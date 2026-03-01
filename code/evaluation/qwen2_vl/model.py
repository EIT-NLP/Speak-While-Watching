"""
Unified Qwen2VLChat model for evaluation across all variants.

Variant dispatch:
  - origin / batch  : standard one-shot generation
  - group / gap / overlap : streaming frame-by-frame with custom RoPE & causal mask
  - interleave      : interleave inference with per-frame token generation
"""
from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
import math
import logging

import torch
import numpy as np

from .base import BaseModel
from .prompt import Qwen2VLPromptMixin
from .util import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
CODE_ROOT = Path(__file__).resolve().parents[2]  # .../code/
MODEL_CODE_DIR = CODE_ROOT / "model_code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(MODEL_CODE_DIR))

# ---------------------------------------------------------------------------
# Variant from env
# ---------------------------------------------------------------------------
VARIANT = os.getenv("QWEN2_5_VL_VARIANT", "origin").lower()

# ---------------------------------------------------------------------------
# Attention implementation: flash_attention_2 if available, else sdpa (faster than eager, no extra install)
# ---------------------------------------------------------------------------
try:
    import flash_attn  # noqa: F401
    _ATTN_IMPLEMENTATION = "flash_attention_2"
except ImportError:
    _ATTN_IMPLEMENTATION = "sdpa"



def get_rope_index_25(
    spatial_merge_size: int = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1],
                                  dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, ids in enumerate(total_input_ids):
            ids = ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
            vision_tokens = ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens and remain_images > 0) else len(input_tokens) + 1
                ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens and remain_videos > 0) else len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    second_per_grid_t = 0
                    image_index += 1; remain_images -= 1; ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
                    video_index += 1; remain_videos -= 1; ed = ed_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
                t_index = (range_tensor * second_per_grid_t * 2).long().flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# Helper: extract_vision_info (used by interleave)
# ---------------------------------------------------------------------------
def extract_vision_info(conversations):
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if "image" in ele or "image_url" in ele or "video" in ele or ele.get("type", "") in ("image", "image_url", "video"):
                        vision_infos.append(ele)
    return vision_infos


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}
    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1
    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


# =========================================================================
# Unified Qwen2VLChat
# =========================================================================
class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        self.variant = VARIANT
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            from transformers import AutoProcessor
            from model_code.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            processor_path = os.getenv("QWEN2_5_VL_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.processor = AutoProcessor.from_pretrained(processor_path, padding_side="right")

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems else -1
        assert max_gpu_mem > 0

        if '72b' in self.model_path.lower() or '32b' in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map=split_model(), attn_implementation=_ATTN_IMPLEMENTATION)
            self.model.eval()
        elif auto_split_flag():
            assert world_size == 1
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation=_ATTN_IMPLEMENTATION)
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu', attn_implementation=_ATTN_IMPLEMENTATION)
            self.model.cuda().eval()

        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Shared: prepare content
    # -----------------------------------------------------------------
    def _prepare_content(self, inputs, dataset=None):
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    # -----------------------------------------------------------------
    # Shared: post-process boxed
    # -----------------------------------------------------------------
    def _postprocess_boxed(self, response):
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            counter = 1
            for i, ch in enumerate(resp):
                if ch == '{':
                    counter += 1
                elif ch == '}':
                    counter -= 1
                if counter == 0:
                    return resp[:i]
        return response

    # =================================================================
    # DISPATCH
    # =================================================================
    def generate_inner(self, message, video_start, video_end, dataset=None, answer_len=None):
        if self.variant in ("group", "gap", "overlap"):
            return self._generate_group(message, video_start, video_end, dataset, answer_len)
        elif self.variant == "interleave":
            return self._generate_interleave(message, video_start, video_end, dataset)
        else:
            # origin / batch
            return self._generate_origin(message, video_start, video_end, dataset)

    # =================================================================
    # ORIGIN / BATCH
    # =================================================================
    def _generate_origin(self, message, video_start, video_end, dataset=None):
        try:
            sys.path.append(str(CODE_ROOT / "qwen-vl-utils" / "src"))
            from qwen_vl_utils_forgen import process_vision_info_forgen
        except Exception as err:
            logging.critical("qwen_vl_utils not found")
            raise err

        messages = []
        self.system_prompt = "You are a helpful assistant."
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info_forgen([messages], video_start, video_end)
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')

        generated_ids = self.model.generate(**inputs, repetition_penalty=1.15, max_new_tokens=300)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        out = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = out[0]
        return self._postprocess_boxed(response)

    # =================================================================
    # GROUP / GAP / OVERLAP  (streaming frame-by-frame)
    # =================================================================
    def _generate_group(self, message, video_start, video_end, dataset=None, answer_len=None):
        self.dataset = dataset
        try:
            sys.path.append(str(CODE_ROOT / "qwen-vl-utils" / "src"))
            from qwen_vl_utils_forgen import process_vision_info_forgen
        except Exception as err:
            logging.critical("qwen_vl_utils not found")
            raise err

        messages = []
        self.system_prompt = "You are a helpful assistant."
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info_forgen([messages], video_start, video_end)
        inputs_in = self.processor(text=text[0][:-22], images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
        running_text = text[0][-22:]
        inputs_out = self.processor(text=running_text, images=None, videos=None, return_tensors='pt').to("cuda")

        # Left side: video prefix uses 3D RoPE positions.
        # Right side: assistant text prompt without video_grid_thw, so get_rope_index_25 uses the
        # fallback branch with scalar positions 0,1,2,... matching generation-start-at-0 behavior.
        pos_in, _ = get_rope_index_25(2, inputs_in.input_ids, video_grid_thw=inputs_in.data['video_grid_thw'])
        pos_out, _ = get_rope_index_25(2, inputs_out.input_ids)

        t = inputs_in.data['video_grid_thw'][0][0]
        token_num_pre_frame = inputs_in.data['video_grid_thw'][0][1] / 2 * inputs_in.data['video_grid_thw'][0][2] / 2

        VISION_START = 151652
        VISION_END = 151653
        vision_start_pos = inputs_in.input_ids[0].eq(VISION_START).nonzero(as_tuple=True)[0].item()
        vision_end_pos = inputs_in.input_ids[0].eq(VISION_END).nonzero(as_tuple=True)[0].item()

        # Build video embeds
        pixel_values_videos = inputs_in["pixel_values_videos"].type(self.model.visual.dtype)
        video_grid_thw = inputs_in["video_grid_thw"]
        inputs_embeds_in = self.model.model.embed_tokens(inputs_in.input_ids)
        inputs_embeds_out = self.model.model.embed_tokens(inputs_out.input_ids)
        input_ids_out = inputs_out.input_ids
        video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)

        mask = inputs_in.input_ids == self.model.config.video_token_id
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds_in)
        video_mask = mask_expanded.to(inputs_embeds_in.device)
        video_embeds = video_embeds.to(inputs_embeds_in.device, inputs_embeds_in.dtype)
        inputs_embeds_in = inputs_embeds_in.masked_scatter(video_mask, video_embeds)

        token_schedule = []
        all_tokens = []

        for i in range(t):
            if i != t - 1:
                end_idx = int(vision_start_pos + (i + 1) * token_num_pre_frame + 1)
                temp_ids_in = inputs_in.input_ids[:, :end_idx]
                temp_pos_in = pos_in[:, :, :end_idx]
                temp_embeds_in = inputs_embeds_in[:, :end_idx, :]
            else:
                temp_ids_in = inputs_in.input_ids
                temp_pos_in = pos_in
                temp_embeds_in = inputs_embeds_in

            inputs_embed = torch.cat([temp_embeds_in, inputs_embeds_out], dim=1)
            input_ids = torch.cat([temp_ids_in, input_ids_out], dim=1)
            pos = torch.cat([temp_pos_in, pos_out], dim=2)

            if dataset and "2plus" in dataset:
                frame_answer_tokens = random.randint((answer_len // t // 2), answer_len // t) if answer_len else 3
            else:
                frame_answer_tokens = 3

            token_schedule.append(frame_answer_tokens)
            max_new = 300 if i == t - 1 else frame_answer_tokens

            outputs = self.model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embed,
                position_ids=pos,
                return_dict_in_generate=True,
                max_new_tokens=max_new,
                repetition_penalty=1.15,
                video_grid_thw=video_grid_thw,
                token_len_schedule=token_schedule,
            )

            new_tokens = outputs.sequences[0, input_ids.shape[1]:]
            all_tokens.append(new_tokens)

            new_token_embeds = self.model.model.embed_tokens(new_tokens.unsqueeze(0))
            last_pos_id = pos_out[0, 0, -1]
            new_pos_out, _ = get_rope_index_25(2, new_tokens.unsqueeze(0))
            new_pos_out = new_pos_out.add(last_pos_id + 1)
            inputs_embeds_out = torch.cat([inputs_embeds_out, new_token_embeds], dim=1)
            input_ids_out = torch.cat([input_ids_out, new_tokens.unsqueeze(0)], dim=1)
            pos_out = torch.cat([pos_out, new_pos_out], dim=2)

        all_tokens_tensor = torch.cat(all_tokens, dim=0)
        response = self.processor.tokenizer.decode(all_tokens_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return self._postprocess_boxed(response)

    # =================================================================
    # INTERLEAVE  (per-frame interleave inference)
    # =================================================================
    @torch.no_grad()
    def _interleave_inference(self, inputs, video_start, video_end, frame_answer_tokens=1):
        IM_START = 151644
        IM_END = 151645
        VISION_START = 151652
        VISION_END = 151653
        VIDEO_TOKEN = 151656
        ASSISTANT_ID = 77091
        END_TOKENS = [VISION_END, IM_END, 198]

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        pixel_values_videos = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]
        device = input_ids.device

        if input_ids.shape[0] != 1:
            raise ValueError("Only support batch size = 1")

        pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
        video_embeds_full = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)

        video_token_mask = (input_ids == VIDEO_TOKEN).squeeze(0)
        if video_token_mask.sum().item() != video_embeds_full.shape[0]:
            raise ValueError(f"Video features mismatch: tokens {video_token_mask.sum().item()} / embeds {video_embeds_full.shape[0]}")

        def replace_video_tokens(ids_slice, base_embeds, vid_ptr):
            mask = (ids_slice == VIDEO_TOKEN)
            n = mask.sum().item()
            if n == 0:
                return base_embeds, vid_ptr
            emb_slice = base_embeds.clone()
            emb_slice[mask] = video_embeds_full[vid_ptr:vid_ptr + n].to(base_embeds.dtype)
            return emb_slice, vid_ptr + n

        t, h_orig, w_orig = video_grid_thw[0]
        h = int(h_orig // 2)
        w = int(w_orig // 2)
        frame_tokens = h * w

        vision_start_idx = (input_ids == VISION_START).nonzero()[0][1].item()
        first_video_end_idx = vision_start_idx + frame_tokens + 1
        answer_start_idx = (input_ids == ASSISTANT_ID).nonzero()[0][1].item() - 1
        bos_ids = input_ids[:, answer_start_idx:answer_start_idx + 3]
        total_video_token_len = vision_start_idx + frame_tokens * t
        vid_ptr = 0

        # q + v1
        ids_cur = input_ids[:, :first_video_end_idx]
        embeds_cur_base = self.model.model.embed_tokens(ids_cur)
        embeds_cur, vid_ptr = replace_video_tokens(ids_cur.squeeze(0), embeds_cur_base.squeeze(0), vid_ptr)
        embeds_cur = embeds_cur.unsqueeze(0)

        pos_cur = position_ids[:, :, :ids_cur.shape[1]]
        cur_scalar_pos = pos_cur.max().item() + 1
        cur_t = 2
        cur_video_start_idx = first_video_end_idx
        all_gen_tokens: List[torch.Tensor] = []

        def append_generated(gen_ids):
            nonlocal ids_cur, embeds_cur, pos_cur, cur_scalar_pos
            ids_cur = torch.cat([ids_cur, gen_ids], dim=1)
            emb_gen = self.model.model.embed_tokens(gen_ids)
            embeds_cur = torch.cat([embeds_cur, emb_gen], dim=1)
            new_pos = torch.arange(gen_ids.shape[1], device=device).reshape(1, 1, -1) + cur_scalar_pos
            pos_cur = torch.cat([pos_cur, new_pos.repeat(3, 1, 1)], dim=2)
            cur_scalar_pos += gen_ids.shape[1]

        # a1: BOS + generate
        ids_cur = torch.cat([ids_cur, bos_ids], dim=1)
        bos_embeds = self.model.model.embed_tokens(bos_ids)
        embeds_cur = torch.cat([embeds_cur, bos_embeds], dim=1)
        pos_bos = (torch.arange(bos_ids.shape[-1], device=device).reshape(1, 1, -1) + cur_scalar_pos).repeat(3, 1, 1)
        pos_cur = torch.cat([pos_cur, pos_bos], dim=2)
        cur_scalar_pos += bos_ids.shape[-1]

        outputs = self.model.generate(
            inputs_embeds=embeds_cur, position_ids=pos_cur,
            max_new_tokens=frame_answer_tokens, repetition_penalty=1.15,
            return_dict_in_generate=True, output_attentions=True)
        new_answer = outputs["sequences"]
        all_gen_tokens.append(new_answer)
        append_generated(new_answer)

        # Interleave loop
        while cur_video_start_idx < total_video_token_len:
            next_video_ids = input_ids[:, cur_video_start_idx:cur_video_start_idx + frame_tokens]
            next_base_embeds = self.model.model.embed_tokens(next_video_ids)
            next_embeds, vid_ptr = replace_video_tokens(next_video_ids.squeeze(0), next_base_embeds.squeeze(0), vid_ptr)
            next_embeds = next_embeds.unsqueeze(0)

            t_pos = torch.full((1, 1, frame_tokens), cur_t + cur_scalar_pos, device=device, dtype=torch.long)
            h_idx = torch.arange(h, device=device).repeat_interleave(w) + cur_scalar_pos
            w_idx = torch.arange(w, device=device).repeat(h) + cur_scalar_pos
            hw_pos = torch.stack([h_idx, w_idx], dim=0).reshape(2, -1).unsqueeze(1)
            pos_video = torch.cat([t_pos, hw_pos], dim=0)

            ids_cur = torch.cat([ids_cur, next_video_ids], dim=1)
            embeds_cur = torch.cat([embeds_cur, next_embeds], dim=1)
            pos_cur = torch.cat([pos_cur, pos_video], dim=2)
            cur_scalar_pos = pos_cur.max().item() + 1
            cur_t += 2
            cur_video_start_idx += frame_tokens

            is_last_frame = cur_video_start_idx >= total_video_token_len
            if is_last_frame:
                end_ids = torch.tensor(END_TOKENS, device=device, dtype=input_ids.dtype).reshape(1, -1)
                end_embeds = self.model.model.embed_tokens(end_ids)
                ids_cur = torch.cat([ids_cur, end_ids], dim=1)
                embeds_cur = torch.cat([embeds_cur, end_embeds], dim=1)
                new_pos = (torch.arange(end_ids.shape[1], device=device).reshape(1, 1, -1) + cur_scalar_pos).repeat(3, 1, 1)
                pos_cur = torch.cat([pos_cur, new_pos], dim=2)
                cur_scalar_pos += end_ids.shape[1]

            max_tok = 300 - int(video_end - video_start) * frame_answer_tokens + 3 if is_last_frame else frame_answer_tokens
            outputs = self.model.generate(
                inputs_embeds=embeds_cur, position_ids=pos_cur,
                max_new_tokens=max_tok, repetition_penalty=1.15,
                return_dict_in_generate=True, output_attentions=True)
            new_answer = outputs["sequences"]
            all_gen_tokens.append(new_answer)
            append_generated(new_answer)

        final_generated = torch.cat(all_gen_tokens, dim=1)
        return {"input_ids": ids_cur, "position_ids": pos_cur, "generated_answer": final_generated}

    def _generate_interleave(self, message, video_start, video_end, dataset=None):
        try:
            sys.path.append(str(CODE_ROOT / "qwen-vl-utils" / "src"))
            from qwen_vl_utils_forgen import process_vision_info_forgen
        except Exception as err:
            logging.critical("qwen_vl_utils not found")
            raise err

        messages = []
        self.system_prompt = "You are a helpful assistant."
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info_forgen([messages], video_start, video_end)
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
        position_ids, _ = get_rope_index_25(2, inputs.input_ids, video_grid_thw=inputs.data['video_grid_thw'])
        inputs['position_ids'] = position_ids

        ans = self._interleave_inference(inputs=inputs, video_start=video_start, video_end=video_end)
        new_tokens = ans['generated_answer'][0]
        response = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return self._postprocess_boxed(response)
