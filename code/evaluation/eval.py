"""
Unified evaluation script for all variants (PE-Video).

Usage:
    # origin / batch (standard one-shot)
    QWEN2_5_VL_VARIANT=origin python eval.py --model-path /path/to/model

    # group / gap / overlap (streaming frame-by-frame)
    QWEN2_5_VL_VARIANT=group python eval.py --model-path /path/to/model

    # interleave (per-frame interleave)
    QWEN2_5_VL_VARIANT=interleave python eval.py --model-path /path/to/model
"""
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dataset_utils import load_dataset_livesports, dump_image, dump_video
from qwen2_vl.model import Qwen2VLChat

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars)
# ---------------------------------------------------------------------------
EVAL_ROOT = Path(__file__).resolve().parent          # .../code/evaluation/
CODE_ROOT = EVAL_ROOT.parent                          # .../code/
PROJECT_ROOT = CODE_ROOT.parent
DATASET_ROOT = Path(
    os.getenv("QWEN2_5_VL_DATASET_ROOT", str(PROJECT_ROOT / "dataset"))
)

DEFAULT_OUTPUT_DIR = os.getenv("QWEN2_5_VL_EVAL_OUTPUT_DIR", str(EVAL_ROOT / "output"))
DEFAULT_MODEL_PATH = os.getenv(
    "QWEN2_5_VL_EVAL_MODEL_PATH",
    str(CODE_ROOT / "qwen-vl-finetune" / "output"),
)
DEFAULT_DATA_DIR = os.getenv(
    "QWEN2_5_VL_EVAL_DATA_DIR",
    str(DATASET_ROOT / "test_3_5.jsonl"),
)
DEFAULT_IMG_ROOT = os.getenv(
    "QWEN2_5_VL_EVAL_IMG_ROOT",
    str(DATASET_ROOT / "PE-Video" / "videos" / "test"),
)
VARIANT = os.getenv("QWEN2_5_VL_VARIANT", "origin").lower()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(args):
    data = load_dataset_livesports(json_path=args.data_dir, video_root=args.img_root)

    dump_image_func = lambda line: dump_image(line, args.img_root)
    dump_video_func = lambda line: dump_video(line, args.img_root)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # CoT prompt
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt or (
            " If you are uncertain or the problem is too complex, make a reasoned guess "
            "based on the information provided. Avoid repeating steps indefinitely—provide "
            "your best guess even if unsure."
        )
        print(f"Using CoT prompt: {cot_prompt}")

    print(f"Loading model from {args.model_path}  (variant={args.variant})")
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        use_custom_prompt=True,
        min_pixels=784,
        max_pixels=50176,
    )
    model.set_dump_image(dump_image_func)
    model.set_dump_video(dump_video_func)

    # For group/gap/overlap we may need answer_len from human caption
    need_answer_len = args.variant in ("group", "gap", "overlap")
    processor = None
    if need_answer_len:
        from transformers import AutoProcessor
        processor_path = os.getenv("QWEN2_5_VL_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
        processor = AutoProcessor.from_pretrained(processor_path, padding_side="right")

    results = []
    for i in tqdm(range(len(data)), desc="Running inference"):
        if i >= args.max_count:
            break
        line = data[i]
        index = line['video_id']

        # Ensure JSON serializability
        for k, v in line.items():
            if isinstance(v, np.integer):
                line[k] = int(v)
            elif isinstance(v, np.floating):
                line[k] = float(v)

        messages = model.build_prompt(line, args.dataset)

        if args.use_cot and len(messages) > 0 and messages[-1]['type'] == 'text':
            messages[-1]['value'] += cot_prompt

        # Compute answer_len for group variants
        answer_len = None
        if need_answer_len and processor is not None:
            inputs_tok = processor(text=line['caption'], padding=True, return_tensors='pt')
            answer_len = inputs_tok.input_ids.shape[1]

        response = model.generate(messages, line['video_start'], line['video_end'],
                                  args.dataset, answer_len)

        print(f"response: {response}")
        print(f"annotation: {line['caption']}")
        print('-' * 50)

        result = {
            "video_id": int(index) if isinstance(index, np.integer) else index,
            "annotation": line["caption"],
            "task": args.dataset,
            "result": response,
            "messages": messages,
        }
        results.append(result)

        if i % 10 == 0:
            with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')

    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

    print(f"Inference completed. Results saved to {args.output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified PE-Video Evaluation Script")
    parser.add_argument("--variant", type=str, default=VARIANT,
                        choices=["origin", "batch", "group", "gap", "overlap", "interleave"],
                        help="Which variant to evaluate")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=str, default="3_5", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to test JSONL file")
    parser.add_argument("--output-file", type=str,
                        default=str(Path(DEFAULT_OUTPUT_DIR) / "eval_infer.json"))
    parser.add_argument("--img-root", type=str, default=DEFAULT_IMG_ROOT,
                        help="Video root directory")
    parser.add_argument("--max-count", type=int, default=400,
                        help="Max number of samples to evaluate (use 1 for quick debug)")
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--cot-prompt", type=str, default="")

    args = parser.parse_args()

    # Sync variant to env so model.py picks it up
    os.environ['QWEN2_5_VL_VARIANT'] = args.variant
    os.environ['LMUData'] = args.data_dir

    run_inference(args)


if __name__ == "__main__":
    main()
