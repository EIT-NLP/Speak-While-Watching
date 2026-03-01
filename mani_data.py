import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_dataset_root = project_root / "dataset"
    default_raw_root = default_dataset_root / "PE-Video"

    parser = argparse.ArgumentParser(
        description=(
            "Prepare PE-Video data in project-local dataset/, then filter by "
            "token-duration ratio and save train_3_5.jsonl/test_3_5.jsonl."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(default_dataset_root),
        help="Project-local dataset root directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="facebook/PE-Video",
        help="HuggingFace dataset id to download when local data is missing.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(default_raw_root / "train"),
        help="Directory containing train split *.json files.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(default_raw_root / "test"),
        help="Directory containing test split *.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_dataset_root),
        help="Output directory for train_3_5.jsonl and test_3_5.jsonl.",
    )
    parser.add_argument(
        "--train-output-name",
        type=str,
        default="train_3_5.jsonl",
        help="Output file name for train split.",
    )
    parser.add_argument(
        "--test-output-name",
        type=str,
        default="test_3_5.jsonl",
        help="Output file name for test split.",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Tokenizer model id used for token counting.",
    )
    return parser.parse_args()


def load_tokenizer(model_id: str):
    print(f"[info] load tokenizer from: {model_id}")
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def has_local_split_jsons(split_dir: Path) -> bool:
    return split_dir.exists() and any(split_dir.glob("*.json"))


def ensure_local_pe_video_jsons(dataset_root: Path, dataset_name: str, train_dir: Path, test_dir: Path) -> None:
    if has_local_split_jsons(train_dir) and has_local_split_jsons(test_dir):
        print("[info] local PE-Video jsons found, skip download.")
        return

    print(f"[info] local jsons missing, downloading {dataset_name} from HuggingFace...")
    cache_dir = dataset_root / "hf_cache"
    dataset_dict = load_dataset(dataset_name, cache_dir=str(cache_dir))

    for split_name in ("train", "test"):
        if split_name not in dataset_dict:
            continue
        out_dir = train_dir if split_name == "train" else test_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, sample in enumerate(tqdm(dataset_dict[split_name], desc=f"Save {split_name} json")):
            video_id = sample.get("video_id", idx)
            file_stem = str(video_id)
            out_file = out_dir / f"{file_stem}.json"
            if out_file.exists():
                out_file = out_dir / f"{file_stem}_{idx}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False)


def ensure_video_path(data: Dict, split_name: str, json_file: str) -> None:
    # Open-source friendly fallback path under project dataset/.
    if "video_path" in data and data["video_path"]:
        return
    video_id = data.get("video_id")
    if video_id is None:
        try:
            video_id = int(Path(json_file).stem)
        except Exception:
            return
    data["video_path"] = f"dataset/PE-Video/videos/{split_name}/{video_id}.mp4"


def merge_jsons_and_filter(video_json_dir: str, split_name: str, tokenizer):
    json_files = glob.glob(os.path.join(video_json_dir, "*.json"))
    print(f"[{split_name}] Found {len(json_files)} json files.")

    output_lines: List[str] = []

    for json_file in tqdm(json_files, desc=f"Processing {split_name}"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            duration = data.get("video_duration_in_s", None)
            if duration is None:
                continue
            duration = float(duration)
            if duration < 5 or duration > 30:
                continue

            caption = data.get("human_caption", "").strip()
            if caption == "":
                continue
            if not caption.lower().startswith("the video shows"):
                continue

            tokenized = tokenizer(caption, add_special_tokens=False)
            token_len = len(tokenized["input_ids"])
            ratio = token_len / duration

            if 3 <= ratio <= 5:
                ensure_video_path(data, split_name=split_name, json_file=json_file)
                output_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return output_lines


def write_jsonl(output_lines: List[str], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(output_lines)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    dataset_root.mkdir(parents=True, exist_ok=True)
    ensure_local_pe_video_jsons(
        dataset_root=dataset_root,
        dataset_name=args.dataset_name,
        train_dir=train_dir,
        test_dir=test_dir,
    )

    tokenizer = load_tokenizer(args.tokenizer_model_id)

    train_lines = merge_jsons_and_filter(
        video_json_dir=str(train_dir),
        split_name="train",
        tokenizer=tokenizer,
    )
    test_lines = merge_jsons_and_filter(
        video_json_dir=str(test_dir),
        split_name="test",
        tokenizer=tokenizer,
    )

    train_output = str(output_dir / args.train_output_name)
    test_output = str(output_dir / args.test_output_name)
    write_jsonl(train_lines, train_output)
    write_jsonl(test_lines, test_output)

    print(f"[done] train output: {train_output} | samples: {len(train_lines)}")
    print(f"[done] test output : {test_output} | samples: {len(test_lines)}")


if __name__ == "__main__":
    main()
