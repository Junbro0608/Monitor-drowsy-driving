#!/usr/bin/env python3
"""
Export a YOLOv8n-cls eye-state model to TensorRT FP32/FP16/INT8 engines on
Jetson and evaluate the exported engines.

Important:
  - Run this on the Jetson Orin Nano, not on Windows/Colab.
  - Do not pass an ORT INT8 ONNX model to TensorRT.
  - FP32 engine is the TensorRT baseline for Jetson-side comparison.
  - INT8 is created by TensorRT calibration from the FP32 .pt model.

Expected folder:
  model_roll_aug_best.onnx
  eye_data_set/ or roll_aug_data/
    train/awake, train/sleepy
    val/awake, val/sleepy
    test/awake, test/sleepy

Example:
  python3 yolo_trt_roll_aug_export_eval.py \
    --model model_roll_aug_best.onnx \
    --dataset roll_aug_data \
    --imgsz 128 \
    --calib-per-class 800 \
    --skip-int8
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLOSE_CLASS_CANDIDATES = {"sleepy", "close", "closed", "drowsy"}


def existing_path(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def class_names(dataset_root: Path, split: str = "train") -> list[str]:
    split_dir = dataset_root / split
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def is_dataset_root(path: Path) -> bool:
    classes = class_names(path, "train")
    if len(classes) != 2:
        return False
    return all((path / split).exists() and class_names(path, split) == classes for split in ["train", "val", "test"])


def find_dataset_root(path: Path) -> Path:
    if is_dataset_root(path):
        return path
    for child in path.rglob("*"):
        if child.is_dir() and is_dataset_root(child):
            return child
    raise FileNotFoundError(f"Could not find train/val/test class dataset under {path}")


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)


def collect_split(dataset_root: Path, split: str) -> tuple[list[str], np.ndarray, dict[str, int]]:
    classes = class_names(dataset_root, "train")
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    paths: list[str] = []
    labels: list[int] = []
    for cls in classes:
        for path in image_files(dataset_root / split / cls):
            paths.append(str(path))
            labels.append(class_to_idx[cls])
    return paths, np.array(labels, dtype=np.int64), class_to_idx


def close_index(class_to_idx: dict[str, int]) -> int:
    for name, idx in class_to_idx.items():
        if name.lower() in CLOSE_CLASS_CANDIDATES:
            return idx
    return 1


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_idx: int) -> tuple[float, float, float, float]:
    """sklearn 없이 accuracy, positive precision/recall/F1을 계산한다."""
    true_pos = y_true == pos_idx
    pred_pos = y_pred == pos_idx
    tp = int(np.logical_and(true_pos, pred_pos).sum())
    fp = int(np.logical_and(~true_pos, pred_pos).sum())
    fn = int(np.logical_and(true_pos, ~pred_pos).sum())

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


def average_precision_binary(pos_true: np.ndarray, pos_prob: np.ndarray) -> float:
    """Positive class PR-AUC(AP)를 numpy만으로 계산한다."""
    pos_true = pos_true.astype(np.int64)
    positives = int(pos_true.sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-pos_prob, kind="mergesort")
    sorted_true = pos_true[order]
    tp = np.cumsum(sorted_true)
    rank = np.arange(1, len(sorted_true) + 1)
    precision_at_k = tp / rank
    return float((precision_at_k * sorted_true).sum() / positives)


def roc_auc_binary(pos_true: np.ndarray, pos_prob: np.ndarray) -> float:
    """Tie 평균 rank 방식으로 ROC-AUC를 계산한다."""
    pos_true = pos_true.astype(bool)
    n_pos = int(pos_true.sum())
    n_neg = int((~pos_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(pos_prob, kind="mergesort")
    sorted_scores = pos_prob[order]
    ranks = np.empty(len(pos_prob), dtype=np.float64)

    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[pos_true].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def make_calibration_dataset(dataset_root: Path, output_dir: Path, per_class: int, seed: int) -> Path:
    rng = random.Random(seed)
    classes = class_names(dataset_root, "train")
    if len(classes) != 2:
        raise ValueError(f"Expected two classes, got {classes}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Ultralytics/TensorRT validates a classification dataset layout. Use the
    # same balanced calibration set for train and val to keep the export simple.
    for split in ["train", "val"]:
        for cls in classes:
            dst = output_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            src_files = image_files(dataset_root / "train" / cls)
            if not src_files:
                raise FileNotFoundError(f"No calibration images found for class={cls}")
            selected = src_files if len(src_files) <= per_class else rng.sample(src_files, per_class)
            for src in selected:
                shutil.copy2(src, dst / src.name)

    return output_dir


def export_engine(
    model_path: Path,
    imgsz: int,
    batch: int,
    out_name: str,
    export_dir: Path,
    **export_kwargs,
) -> Path:
    target = (export_dir / out_name).resolve()
    if target.exists():
        print("Reuse existing engine:", target)
        return target

    if model_path.suffix.lower() == ".onnx":
        target.parent.mkdir(parents=True, exist_ok=True)

        trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
        cmd = [
            trtexec,
            f"--onnx={model_path}",
            f"--saveEngine={target}",
        ]
        if export_kwargs.get("half"):
            cmd.append("--fp16")
        if export_kwargs.get("int8"):
            cmd.append("--int8")

        subprocess.run(cmd, check=True)
        return target

    model = YOLO(str(model_path), task="classify")
    exported = Path(
        model.export(
            format="engine",
            imgsz=imgsz,
            batch=batch,
            device=0,
            simplify=False,
            **export_kwargs,
        )
    ).resolve()
    if exported != target:
        shutil.copy2(exported, target)
    return target


def evaluate_model(
    model_path: Path,
    dataset_root: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: int | str,
    max_images: int | None = None,
) -> dict:
    model = YOLO(str(model_path), task="classify")
    paths, y_true, class_to_idx = collect_split(dataset_root, split)
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
        y_true = y_true[:max_images]
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in split={split}")

    probs = []
    for i in range(0, len(paths), batch):
        results = model(paths[i : i + batch], imgsz=imgsz, device=device, verbose=False)
        probs.extend(r.probs.data.detach().cpu().numpy() for r in results)

    prob_arr = np.vstack(probs)
    y_pred = prob_arr.argmax(axis=1)
    pos_idx = close_index(class_to_idx)
    pos_true = (y_true == pos_idx).astype(np.int64)
    pos_prob = prob_arr[:, pos_idx]
    accuracy, precision, recall, f1 = binary_metrics(y_true, y_pred, pos_idx)

    metrics = {
        "images": int(len(paths)),
        "accuracy": accuracy,
        "close_precision": float(precision),
        "close_recall": float(recall),
        "f1": float(f1),
        "pr_auc": average_precision_binary(pos_true, pos_prob),
    }
    metrics["roc_auc"] = roc_auc_binary(pos_true, pos_prob)
    return metrics


def benchmark_model(
    model_path: Path,
    dataset_root: Path,
    imgsz: int,
    device: int | str,
    warmup: int,
    count: int,
) -> dict:
    model = YOLO(str(model_path), task="classify")
    paths, _, _ = collect_split(dataset_root, "test")
    if not paths:
        raise FileNotFoundError("No test images found for benchmark")
    needed = max(1, warmup + count)
    paths = (paths * ((needed // len(paths)) + 1))[:needed]

    for path in paths[:warmup]:
        model(path, imgsz=imgsz, device=device, verbose=False)

    latencies = []
    for path in paths[warmup : warmup + count]:
        start = time.perf_counter()
        model(path, imgsz=imgsz, device=device, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(latencies, dtype=np.float64)
    return {
        "benchmark_images": int(count),
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "fps_mean": float(1000.0 / arr.mean()) if arr.mean() > 0 else float("nan"),
    }


def write_html(summary: dict, output_path: Path) -> None:
    rows = []
    for name in ["fp32", "fp16", "int8"]:
        if f"{name}_engine" not in summary:
            continue
        metrics = summary.get(f"{name}_test_metrics", {})
        bench = summary.get(f"{name}_benchmark", {})
        rows.append(
            "<tr>"
            f"<td>{name.upper()}</td>"
            f"<td>{summary.get(f'{name}_engine_bytes', 0) / (1024 * 1024):.2f}</td>"
            f"<td>{metrics.get('accuracy', float('nan')):.6f}</td>"
            f"<td>{metrics.get('f1', float('nan')):.6f}</td>"
            f"<td>{metrics.get('close_recall', float('nan')):.6f}</td>"
            f"<td>{metrics.get('pr_auc', float('nan')):.6f}</td>"
            f"<td>{bench.get('latency_ms_mean', float('nan')):.3f}</td>"
            f"<td>{bench.get('fps_mean', float('nan')):.2f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>YOLO TensorRT Engine Benchmark Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f4f4f5; }}
code {{ background: #f4f4f5; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>YOLO TensorRT Engine Benchmark Report</h1>
<p>Model: <code>{summary.get('model')}</code></p>
<p>Dataset: <code>{summary.get('dataset')}</code></p>
<p>Calibration dataset: <code>{summary.get('calibration_dataset', '-')}</code></p>
<table>
<thead>
<tr>
<th>Engine</th><th>Size MB</th><th>Accuracy</th><th>F1</th><th>Close Recall</th><th>PR-AUC</th><th>Mean Latency ms</th><th>FPS</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<h2>Note</h2>
<p>FP32 is the TensorRT baseline engine. FP16 is the deployment candidate for Jetson speed comparison. INT8 is optional and should be used only when calibration and compatibility are confirmed.</p>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="model_roll_aug_best.onnx",
        help="YOLOv8n-cls .onnx or .pt model. Use .onnx on offline Jetson to avoid Python onnx dependency.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset root or parent folder")
    parser.add_argument("--output-dir", default="trt_roll_aug_quantization")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1, help="Engine batch. Use 1 for camera deployment.")
    parser.add_argument("--eval-batch", type=int, default=1, help="Use 1 when engine batch is fixed to 1.")
    parser.add_argument("--calib-per-class", type=int, default=800)
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace in GiB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-fp32", action="store_true")
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--fp32-engine", default="", help="Existing FP32 TensorRT engine path. Skips FP32 export.")
    parser.add_argument("--fp16-engine", default="", help="Existing FP16 TensorRT engine path. Skips FP16 export.")
    parser.add_argument("--int8-engine", default="", help="Existing INT8 TensorRT engine path. Skips INT8 export.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--eval-max-images", type=int, default=0, help="0 means full test split")
    parser.add_argument("--benchmark-images", type=int, default=300)
    parser.add_argument("--warmup-images", type=int, default=30)
    args = parser.parse_args()

    model_path = existing_path(args.model)
    dataset_root = find_dataset_root(existing_path(args.dataset))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_root = None
    if not args.skip_int8:
        calib_root = make_calibration_dataset(dataset_root, output_dir / "calib_dataset", args.calib_per_class, args.seed)

    print("Model:", model_path)
    print("Dataset:", dataset_root)
    print("Output:", output_dir)
    print("Calibration dataset:", calib_root if calib_root is not None else "-")

    summary = {
        "model": str(model_path),
        "dataset": str(dataset_root),
        "output_dir": str(output_dir),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "calib_per_class": args.calib_per_class,
    }
    if calib_root is not None:
        summary["calibration_dataset"] = str(calib_root)

    engine_prefix = model_path.stem
    engines: dict[str, Path] = {}
    if args.fp32_engine:
        engines["fp32"] = existing_path(args.fp32_engine)
        print("Existing FP32 engine:", engines["fp32"])
    elif not args.skip_fp32:
        engines["fp32"] = export_engine(
            model_path=model_path,
            imgsz=args.imgsz,
            batch=args.batch,
            out_name=f"{engine_prefix}_fp32.engine",
            export_dir=output_dir,
            workspace=args.workspace,
        )
        print("FP32 engine:", engines["fp32"])

    if args.fp16_engine:
        engines["fp16"] = existing_path(args.fp16_engine)
        print("Existing FP16 engine:", engines["fp16"])
    elif not args.skip_fp16:
        engines["fp16"] = export_engine(
            model_path=model_path,
            imgsz=args.imgsz,
            batch=args.batch,
            out_name=f"{engine_prefix}_fp16.engine",
            export_dir=output_dir,
            half=True,
            workspace=args.workspace,
        )
        print("FP16 engine:", engines["fp16"])

    if args.int8_engine:
        engines["int8"] = existing_path(args.int8_engine)
        print("Existing INT8 engine:", engines["int8"])
    elif not args.skip_int8:
        engines["int8"] = export_engine(
            model_path=model_path,
            imgsz=args.imgsz,
            batch=args.batch,
            out_name=f"{engine_prefix}_int8.engine",
            export_dir=output_dir,
            int8=True,
            data=str(calib_root),
            fraction=1.0,
            workspace=args.workspace,
        )
        print("INT8 engine:", engines["int8"])

    for name, engine in engines.items():
        summary[f"{name}_engine"] = str(engine)
        summary[f"{name}_engine_bytes"] = engine.stat().st_size

    eval_max = None if args.eval_max_images <= 0 else args.eval_max_images
    if not args.skip_eval:
        for name, engine in engines.items():
            summary[f"{name}_test_metrics"] = evaluate_model(
                engine, dataset_root, "test", args.imgsz, args.eval_batch, 0, eval_max
            )

    if not args.skip_benchmark:
        for name, engine in engines.items():
            summary[f"{name}_benchmark"] = benchmark_model(
                engine, dataset_root, args.imgsz, 0, args.warmup_images, args.benchmark_images
            )

    summary_path = output_dir / "trt_engine_benchmark_summary.json"
    html_path = output_dir / "trt_engine_benchmark_report.html"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(summary, html_path)

    print("Summary:", summary_path)
    print("HTML report:", html_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
