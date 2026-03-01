import os
import json
import shutil
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from common_utils import download_file, md5, toliststr, decode_base64_to_image_file


# ---------------------------------------------------------------------------
# PE-Video dataset loader  (default for all variants)
# ---------------------------------------------------------------------------
EVAL_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EVAL_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
DATASET_ROOT = Path(
    os.getenv("QWEN2_5_VL_DATASET_ROOT", str(PROJECT_ROOT / "dataset"))
)
DEFAULT_JSON_PATH = os.getenv(
    "QWEN2_5_VL_EVAL_DATA_DIR",
    str(DATASET_ROOT / "test_3_5.jsonl"),
)
DEFAULT_VIDEO_ROOT = os.getenv(
    "QWEN2_5_VL_EVAL_IMG_ROOT",
    str(DATASET_ROOT / "PE-Video" / "videos" / "test"),
)


def load_dataset_livesports(
    json_path: str = DEFAULT_JSON_PATH,
    video_root: str = DEFAULT_VIDEO_ROOT,
) -> List[Dict[str, str]]:
    """Load PE-Video JSONL dataset for evaluation."""
    samples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                video_id = item['video_id']
                video_path = item['video_path']

                # Support relative paths in open-source setup.
                if not os.path.isabs(video_path):
                    candidate_paths = [
                        video_path,
                        str(PROJECT_ROOT / video_path),
                        os.path.join(video_root, video_path),
                        os.path.join(video_root, os.path.basename(video_path)),
                    ]
                    for c in candidate_paths:
                        if os.path.exists(c):
                            video_path = c
                            break

                if not os.path.exists(video_path):
                    print(f"Warning: Video file {video_path} not found. Skipping.")
                    continue
                if 'mkv' in video_path:
                    print(f"Warning: Video file {video_path} is in MKV format. Skipping.")
                    continue

                samples.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'caption': item.get('human_caption', ''),
                    'video_start': 0,
                    'video_end': item.get('video_duration_in_s', 0),
                })
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    return samples


# ---------------------------------------------------------------------------
# ActivityNet / LLaVA-Video loaders (kept for backward-compat)
# ---------------------------------------------------------------------------
def load_dataset(
    json_path: str = 'dataset/ActivityNet_Captions/activitynet_captions_val1.json',
    video_root: str = 'dataset/ActivityNet_Captions/extracted/video',
) -> List[Dict[str, str]]:
    """Load ActivityNet Captions dataset."""
    assert os.path.exists(json_path), f"JSON file {json_path} not found."
    assert os.path.exists(video_root), f"Video root {video_root} not found."

    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        video_id = item['video_id']
        video_filename = item['video']
        caption = item['caption']
        video_path = os.path.join(video_root, video_filename)
        if not os.path.exists(video_path):
            continue
        if 'mkv' in video_path:
            continue
        samples.append({
            'video_id': video_id,
            'video_path': video_path,
            'prompt': "<video>\nDescribe every scene and its significance in the video.",
            'gt_caption': caption,
        })
    return samples


def load_dataset_llava_video(
    json_path: str = 'dataset/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_valid_eval.json',
    video_root: str = 'dataset/LLaVA-Video-178K/0_30_s_academic_v0_1',
) -> List[Dict[str, str]]:
    """Load LLaVA-Video dataset."""
    assert os.path.exists(json_path), f"JSON file {json_path} not found."
    assert os.path.exists(video_root), f"Video root {video_root} not found."

    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        video_id = item['id']
        video_filename = item['video']
        caption = item['conversations'][1]['value']
        video_path = os.path.join(video_root, video_filename)
        if not os.path.exists(video_path):
            continue
        if 'mkv' in video_path:
            continue
        samples.append({
            'video_id': video_id,
            'video_path': video_path,
            'prompt': item['conversations'][0]['value'],
            'gt_caption': caption,
        })
    return samples


# ---------------------------------------------------------------------------
# Image / video dump helpers
# ---------------------------------------------------------------------------
def dump_image(line, img_root):
    os.makedirs(img_root, exist_ok=True)
    if 'image' in line:
        if isinstance(line['image'], list):
            tgt_path = []
            assert 'image_path' in line
            for img, im_name in zip(line['image'], line['image_path']):
                path = os.path.join(img_root, im_name)
                if not os.path.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = os.path.join(img_root, f"{line['index']}.jpg")
            if not os.path.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path = [tgt_path]
    else:
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
    return tgt_path


def dump_video(line, video_root):
    os.makedirs(video_root, exist_ok=True)
    return [line["video_path"]]


# ---------------------------------------------------------------------------
# MMMU preprocessing
# ---------------------------------------------------------------------------
def MMMU_preproc(data):
    cnt = 0
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            cnt += 1
    print(f'During MMMU_preproc, {cnt} open questions re-formulated to multi-choice.')
    data['A'] = As
    data['B'] = Bs
    return data
