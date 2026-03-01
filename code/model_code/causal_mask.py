import os
import random

import torch

# Special token ids.
IM_START = 151644
IM_END = 151645
VISION_START = 151652
VISION_END = 151653
VIDEO_TOKEN = 151656
ASSISTANT_ID = 77091  # assistant token
SYSTEM_ID = 8948  # system token


def parse_segments(input_ids_1d):
    """Return list of (start, end) for each <|im_start|>...<|im_end|> block."""
    im_start_idxs = (input_ids_1d == IM_START).nonzero(as_tuple=True)[0]
    im_end_idxs = (input_ids_1d == IM_END).nonzero(as_tuple=True)[0]
    assert len(im_start_idxs) == len(im_end_idxs), "Mismatched <|im_start|> and <|im_end|> counts"
    return list(zip(im_start_idxs.tolist(), im_end_idxs.tolist()))


def categorize_segments(input_ids_1d, segments):
    """Return (P, Q, [V], [A]) with assistant_id used for identifying answer blocks"""
    P, Q, V_list, A_list = None, None, [], []
    for (s, e) in segments:
        tokens = input_ids_1d[s : e + 1]
        if VISION_START in tokens and VISION_END in tokens:
            V_list.append((s, e))
        elif SYSTEM_ID in tokens:
            P = (s, e)
        elif ASSISTANT_ID in tokens:
            A_list.append((s, e))
        elif Q is None:
            Q = (s, e)
    return P, Q, V_list, A_list


def build_batch_custom_causal_mask(input_ids: torch.Tensor, dtype=torch.float32):
    """
    Args:
        input_ids: (B, L) tensor
    Returns:
        causal_mask: (B, 1, L, L) tensor, -inf for masked, 0 for visible
    """
    B, L = input_ids.shape
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((B, 1, L, L), min_dtype, device=device)

    for b in range(B):
        ids = input_ids[b]
        segments = parse_segments(ids)
        P, Q, V_list, A_list = categorize_segments(ids, segments)

        # Default upper-triangular mask to prevent information leakage.
        causal_mask[b, 0] = torch.triu(torch.full((L, L), min_dtype, device=device), diagonal=1)

        for i, (a_s, a_e) in enumerate(A_list):
            allowed = list(range(P[0], P[1] + 1)) + list(range(Q[0], Q[1] + 1))
            for v in V_list[: i + 1]:
                allowed += list(range(v[0], v[1] + 1))
            for prev_a in A_list[:i]:
                allowed += list(range(prev_a[0], prev_a[1] + 1))
            for t in range(a_s, a_e + 1):
                allowed.append(t)
                causal_mask[b, 0, t, :] = min_dtype
                causal_mask[b, 0, t, allowed] = 0.0

    return causal_mask


def build_group_custom_causal_mask_for_eval(
    input_ids: torch.Tensor,
    video_frame_token_counts: list,
    ANSWER_GROUP_SIZE=3,
    token_len_schedule=None,
    dtype=torch.float32,
    debug=False,
):
    """
    Causal mask for group/gap/overlap evaluation (aligned with causal_mask_prefill behavior).
    The answer segment is split by token_len_schedule; in prefill, each frame maps to one
    answer chunk.
    """
    B, L = input_ids.shape
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((B, 1, L, L), min_dtype, device=device)

    for b in range(B):
        ids = input_ids[b]
        im_start_idxs = (ids == IM_START).nonzero(as_tuple=True)[0]
        visual_start_idxs = (ids == VISION_START).nonzero(as_tuple=True)[0]
        P = (im_start_idxs[0].item(), im_start_idxs[1].item() - 1)
        QV = (im_start_idxs[1].item(), im_start_idxs[2].item() - 1)
        visual_token_len = im_start_idxs[2].item() - 1 - visual_start_idxs[0].item()
        A = ids[im_start_idxs[2] :]
        A_list = []
        if token_len_schedule is not None and len(token_len_schedule) > 0:
            offset = 3
            for ans_len in token_len_schedule:
                if offset >= len(A):
                    break
                start = im_start_idxs[2].item() + offset
                end = im_start_idxs[2].item() + min(offset + ans_len - 1, len(A) - 1)
                A_list.append((start, end))
                offset += ans_len
        else:
            for i in range(3, len(A), ANSWER_GROUP_SIZE):
                A_list.append(
                    (
                        im_start_idxs[2].item() + i,
                        im_start_idxs[2].item() + min(i + ANSWER_GROUP_SIZE - 1, len(A) - 1),
                    )
                )

        causal_mask[b, 0] = torch.triu(torch.full((L, L), min_dtype, device=device), diagonal=1)
        if len(A_list) < 1:
            continue
        frame_num = int(visual_token_len / video_frame_token_counts[b])
        vision_frame_spans = []
        start_idx = visual_start_idxs[0].item() + 1
        frame_token_count = video_frame_token_counts[b]
        for i in range(frame_num):
            s = start_idx + i * frame_token_count
            e = s + frame_token_count - 1
            vision_frame_spans.append((s, e))
        if not vision_frame_spans:
            continue
        vision_frame_spans[0] = (vision_frame_spans[0][0] - 1, vision_frame_spans[0][1])
        if VISION_END in ids:
            vision_frame_spans[-1] = (vision_frame_spans[-1][0], vision_frame_spans[-1][1] + 3)
        A_list[0] = (A_list[0][0] - 3, A_list[0][1])
        for group_idx, (a_s, a_e) in enumerate(A_list):
            allowed = list(range(P[0], P[1] + 1))
            q_tokens_end = visual_start_idxs[0].item() - 1
            allowed += list(range(QV[0], q_tokens_end + 1))
            for i in range(group_idx + 1):
                if i < len(vision_frame_spans):
                    vs, ve = vision_frame_spans[i]
                    allowed += list(range(vs, ve + 1))
            if group_idx > 0:
                prev_a_start = A_list[0][0]
                allowed += list(range(prev_a_start, a_s))
            for t in range(a_s, a_e + 1):
                allowed.append(t)
                causal_mask[b, 0, t, :] = min_dtype
                causal_mask[b, 0, t, allowed] = 0.0
        if len(A_list) >= 1 and len(vision_frame_spans) >= 2:
            prefix_span = A_list[0][0] + 3
            causal_mask[b, 0, prefix_span:-1, : prefix_span - 3] = causal_mask[
                b, 0, prefix_span + 1 :, : prefix_span - 3
            ].clone()
        causal_mask[b, 0, -1, -1] = 0
    return causal_mask


def build_group_custom_causal_mask_for_train(
    input_ids: torch.Tensor,
    video_frame_token_counts: list,
    ANSWER_GROUP_SIZE=3,
    dtype=torch.float32,
    debug=False,
    dataset_use=None,
):

    B, L = input_ids.shape
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((B, 1, L, L), min_dtype, device=device)

    if dataset_use is not None and "2plus" not in dataset_use:
        use_strict_groups = True
    else:
        use_strict_groups = False

    for b in range(B):
        ids = input_ids[b]
        segments = parse_segments(ids)
        P, QV, A_list = None, None, []
        for (s, e) in segments:
            tokens = ids[s : e + 1]
            if SYSTEM_ID in tokens:
                P = (s, e)
            elif ASSISTANT_ID in tokens:
                A_list.append((s + 3, e))
            elif QV is None:
                QV = (s, e)

        assert P is not None and QV is not None and len(A_list) > 0, "Missing required segments"

        QV_tokens = ids[QV[0] : QV[1] + 1]
        vision_start_pos = QV_tokens.eq(VISION_START).nonzero(as_tuple=True)[0].item()
        vision_end_pos = QV_tokens.eq(VISION_END).nonzero(as_tuple=True)[0].item()
        vision_token_start = QV[0] + vision_start_pos + 1
        vision_token_end = QV[0] + vision_end_pos - 1
        total_video_token_len = vision_token_end - vision_token_start + 1
        N_token_per_frame = video_frame_token_counts[b]
        num_frames = total_video_token_len // N_token_per_frame
        vision_frame_spans = []
        curr = vision_token_start
        for _ in range(num_frames):
            v_s = int(curr)
            v_e = int(curr + N_token_per_frame - 1)
            vision_frame_spans.append((v_s, v_e))
            curr += N_token_per_frame

        if debug:
            print(f"\n=== [Batch {b}] P={P} QV={QV} A_list={A_list} vision_frame_spans={vision_frame_spans}")

        causal_mask[b, 0] = torch.triu(torch.full((L, L), min_dtype, device=device), diagonal=1)

        for i, (a_s, a_e) in enumerate(A_list):
            answer_len = a_e - a_s + 1
            num_vision_frames = len(vision_frame_spans)
            num_answer_groups = num_vision_frames
            if use_strict_groups:
                min_required = ANSWER_GROUP_SIZE * (num_vision_frames - 1) + 1
                if answer_len < min_required:
                    raise ValueError(
                        f"[Batch {b}] Not enough answer tokens! Required ≥ {min_required}, got {answer_len}."
                    )
            cur_pos = a_s
            for group_idx in range(num_answer_groups):
                if group_idx < num_answer_groups - 1:
                    group_size = (
                        ANSWER_GROUP_SIZE
                        if use_strict_groups
                        else random.randint(
                            max(1, (a_e - a_s) // num_vision_frames // 2),
                            max(1, (a_e - a_s) // num_vision_frames),
                        )
                    )
                    g_s = cur_pos
                    g_e = g_s + group_size - 1
                else:
                    g_s = cur_pos
                    g_e = a_e
                allowed = list(range(P[0], P[1] + 1))
                allowed.append(vision_token_start - 1)
                allowed.append(QV[0] - 1)
                q_tokens_end = vision_token_start - 2
                if q_tokens_end >= QV[0]:
                    allowed += list(range(QV[0], q_tokens_end + 1))
                for j in range(group_idx + 1):
                    pv_s, pv_e = vision_frame_spans[j]
                    allowed += list(range(pv_s, pv_e + 1))
                    if j == num_answer_groups - 1:
                        allowed.extend([pv_e + 1, pv_e + 2, pv_e + 3])
                if g_s > a_s:
                    allowed += list(range(a_s, g_s))
                allowed.extend([A_list[0][0] - 3, A_list[0][0] - 2, A_list[0][0] - 1])
                for t in range(g_s, g_e + 1):
                    allowed.append(t)
                    causal_mask[b, 0, t, :] = min_dtype
                    causal_mask[b, 0, t, allowed] = 0.0
                cur_pos = g_e + 1

        # Apply post-adjustments so assistant prefix tokens and answer rows
        # have the intended visibility over the prompt/vision prefix.
        assitant_prefix = vision_frame_spans[1][0]
        prefix_span = QV[1] + 5
        causal_mask[b, 0, prefix_span - 3 : prefix_span, assitant_prefix : prefix_span - 3] = min_dtype
        causal_mask[b, 0, prefix_span:-1, : prefix_span - 3] = causal_mask[
            b, 0, prefix_span + 1 :, : prefix_span - 3
        ]
    return causal_mask


def build_group_custom_causal_mask(
    input_ids: torch.Tensor,
    video_frame_token_counts: list,
    ANSWER_GROUP_SIZE=3,
    dtype=torch.float32,
    debug=False,
    dataset_use=None,
    token_len_schedule=None,
):

    if token_len_schedule is not None and len(token_len_schedule) > 0:
        return build_group_custom_causal_mask_for_eval(
            input_ids=input_ids,
            video_frame_token_counts=video_frame_token_counts,
            ANSWER_GROUP_SIZE=ANSWER_GROUP_SIZE,
            token_len_schedule=token_len_schedule,
            dtype=dtype,
            debug=debug,
        )
    return build_group_custom_causal_mask_for_train(
        input_ids=input_ids,
        video_frame_token_counts=video_frame_token_counts,
        ANSWER_GROUP_SIZE=ANSWER_GROUP_SIZE,
        dtype=dtype,
        debug=debug,
        dataset_use=dataset_use,
    )


def build_inference_causal_mask(
    input_ids: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    config,
    dtype=torch.float32,
):
    """
    Build a causal mask that reflects interleave layout during inference.
    Only applies to the current decoding step (sequence_length == 1).
    """
    batch_size, target_length = attention_mask.shape
    sequence_length = 1
    device = input_ids.device
    min_dtype = torch.finfo(dtype).min

    if past_key_values is not None and hasattr(past_key_values, "get_max_cache_shape"):
        total_kv_len = past_key_values.get_max_cache_shape()
    else:
        total_kv_len = target_length

    causal_mask = torch.full(
        (batch_size, 1, sequence_length, total_kv_len), fill_value=min_dtype, dtype=dtype, device=device
    )

    for b in range(batch_size):
        ids = input_ids[b]
        cache_pos = cache_position[b]
        full_ids = ids[: cache_pos.item() + 1]

        segments = parse_segments(full_ids)
        P, Q, V_list, A_list = categorize_segments(full_ids, segments)

        current_t = cache_pos.item()
        allow_set = set()

        for i, (a_s, a_e) in enumerate(A_list):
            if a_s <= current_t <= a_e:
                allow_set.update(range(P[0], P[1] + 1))
                allow_set.update(range(Q[0], Q[1] + 1))
                for v in V_list[: i + 1]:
                    allow_set.update(range(v[0], v[1] + 1))
                for prev_a in A_list[:i]:
                    allow_set.update(range(prev_a[0], prev_a[1] + 1))
                break
        else:
            allow_set.update(range(0, current_t + 1))

        allow_list = sorted(list(allow_set))
        causal_mask[b, 0, 0, allow_list] = 0.0

    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        padding_mask = causal_mask + attention_mask[:, None, None, :].to(device)
        causal_mask = causal_mask.masked_fill(padding_mask == 0, min_dtype)

    return causal_mask


