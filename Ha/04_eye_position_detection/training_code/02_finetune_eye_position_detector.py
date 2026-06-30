import argparse
import json
import shutil
import zipfile
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MAIN_ROOT = ROOT.parent
DATA_ZIP = ROOT / "cleaned" / "eye_detect_only_cleaned.zip"
PRETRAIN_BEST = MAIN_ROOT / "eye_mixed_awake_sleep" / "outputs" / "mixed_eye_yolov8n" / "weights" / "best.pt"
PREPARED_DIR = ROOT / "converted_eye_1class" / "prepared_eye_detect_only_cleaned"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"


def reset_dir(path: Path) -> None:
    """기존 준비 폴더를 지우고 새로 만든다."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_dataset(zip_path: Path, out_dir: Path) -> Path:
    """YOLO zip 데이터셋을 풀고 data.yaml 경로를 로컬 절대경로로 보정한다."""
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    reset_dir(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"train: {(out_dir / 'train' / 'images').as_posix()}",
                f"val: {(out_dir / 'valid' / 'images').as_posix()}",
                f"test: {(out_dir / 'test' / 'images').as_posix()}",
                "",
                "nc: 1",
                "names: ['eye']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def dataset_summary(data_dir: Path) -> dict:
    """이미지/라벨 개수와 빈 라벨 개수를 확인한다."""
    labels = list(data_dir.glob("**/labels/*.txt"))
    empty_labels = [p for p in labels if not p.read_text(encoding="utf-8").strip()]
    boxes = 0
    classes = {}
    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                classes[parts[0]] = classes.get(parts[0], 0) + 1
                boxes += 1

    return {
        "train_images": len(list((data_dir / "train" / "images").glob("*"))),
        "valid_images": len(list((data_dir / "valid" / "images").glob("*"))),
        "test_images": len(list((data_dir / "test" / "images").glob("*"))),
        "label_files": len(labels),
        "empty_label_files": len(empty_labels),
        "boxes": boxes,
        "class_counts": classes,
    }


def train(args: argparse.Namespace) -> None:
    """mixed 1차 best 모델을 eye-detect-only 데이터셋으로 파인튜닝한다."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pretrain = Path(args.pretrain) if args.pretrain else PRETRAIN_BEST
    if not pretrain.exists() and not args.check_only:
        raise FileNotFoundError(f"Pretrain best.pt not found: {pretrain}")

    data_yaml = extract_dataset(DATA_ZIP, PREPARED_DIR)
    summary = dataset_summary(PREPARED_DIR)
    summary["pretrain"] = str(pretrain)
    summary["pretrain_exists"] = pretrain.exists()
    summary["environment"] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary["data_yaml"] = str(data_yaml)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.check_only:
        return

    model = YOLO(str(pretrain))
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(OUTPUT_DIR),
        name="eye_detect_only_finetune_yolov8n",
        exist_ok=True,
        optimizer="auto",
        lr0=args.lr0,
        cos_lr=True,
        plots=True,
    )

    run_dir = Path(result.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=args.device, plots=True)

    report = {
        "stage": "eye_detect_only_finetune",
        "model": "yolov8n",
        "dataset": "eye_detect_only_cleaned",
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "pretrain": str(pretrain),
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
    }
    (REPORT_DIR / "eye_detect_only_finetune_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.export_onnx:
        best_model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)


def parse_args() -> argparse.Namespace:
    """명령행 파인튜닝 옵션을 정의한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", default=-1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
