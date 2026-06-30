from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from ultralytics import YOLO

from run_yolo_roll50_finetune import (
    class_names,
    collect_split,
    evaluate_paths,
    find_dataset_root,
    image_files,
    rotate_keep_size,
    sample_by_class,
)


def parse_angles(text: str) -> list[int]:
    angles = [int(x.strip()) for x in text.split(",") if x.strip()]
    if 0 not in angles:
        angles.append(0)
    return sorted(set(angles))


def angle_tag(angle: int) -> str:
    if angle < 0:
        return f"m{abs(angle)}"
    if angle > 0:
        return f"p{angle}"
    return "0"


def output_name(src: Path, angle: int) -> str:
    return f"{src.stem}__roll_{angle_tag(angle)}{src.suffix.lower()}"


def write_rotated_image(src: Path, dst: Path, angle: int, force: bool) -> tuple[bool, str | None]:
    if dst.exists() and not force:
        return False, None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if angle == 0:
        shutil.copy2(src, dst)
        return True, None
    img = cv2.imread(str(src))
    if img is None:
        return False, f"Could not read image: {src}"
    rotated = rotate_keep_size(img, float(angle))
    ok = cv2.imwrite(str(dst), rotated)
    if not ok:
        return False, f"Could not write image: {dst}"
    return True, None


def build_roll_dataset(
    source_root: Path,
    output_root: Path,
    angles: list[int],
    build_workers: int,
    force: bool,
) -> dict:
    classes = class_names(source_root, "train")
    if len(classes) != 2:
        raise ValueError(f"Expected two classes under train/, got {classes}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "angles": angles,
        "splits": {},
    }

    if (source_root / "labels.txt").exists():
        shutil.copy2(source_root / "labels.txt", output_root / "labels.txt")

    total_created = 0
    total_skipped = 0
    errors: list[str] = []

    for split in ["train", "val"]:
        manifest["splits"][split] = {}
        jobs = []
        for cls in classes:
            src_files = image_files(source_root / split / cls)
            manifest["splits"][split][cls] = {
                "source_images": len(src_files),
                "angles_per_image": len(angles),
                "expected_output_images": len(src_files) * len(angles),
            }
            for src in src_files:
                for angle in angles:
                    dst = output_root / split / cls / output_name(src, angle)
                    jobs.append((src, dst, angle))

        print(f"[build] {split}: {len(jobs)} output images")
        created = skipped = 0
        with ThreadPoolExecutor(max_workers=max(1, build_workers)) as executor:
            futures = [executor.submit(write_rotated_image, src, dst, angle, force) for src, dst, angle in jobs]
            for i, future in enumerate(as_completed(futures), start=1):
                did_write, error = future.result()
                if did_write:
                    created += 1
                else:
                    skipped += 1
                if error:
                    errors.append(error)
                if i % 10000 == 0 or i == len(futures):
                    print(f"[build] {split}: {i}/{len(futures)} done, created={created}, skipped={skipped}")
        total_created += created
        total_skipped += skipped

    # Keep test split unaugmented in the training dataset root. Separate roll
    # evaluation is done in memory, angle by angle, to avoid mixing metrics.
    for cls in classes:
        test_files = image_files(source_root / "test" / cls)
        manifest["splits"].setdefault("test", {})[cls] = {
            "source_images": len(test_files),
            "angles_per_image": 1,
            "expected_output_images": len(test_files),
        }
        for src in test_files:
            dst = output_root / "test" / cls / src.name
            if dst.exists() and not force:
                total_skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            total_created += 1

    manifest["created_images"] = total_created
    manifest["skipped_existing_images"] = total_skipped
    manifest["errors"] = errors[:100]
    manifest_path = output_root / "roll_aug_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"Dataset build completed with {len(errors)} errors. First error: {errors[0]}")
    return manifest


def train_model(args: argparse.Namespace, dataset_root: Path, output_dir: Path) -> Path:
    model_path = Path(args.model).expanduser().resolve()
    model = YOLO(str(model_path), task="classify")
    run_project = output_dir.parent
    model.train(
        data=str(dataset_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(run_project),
        name=args.run_name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        degrees=0.0,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        erasing=0.0,
        auto_augment=None,
        cache=False,
        seed=args.seed,
        patience=args.patience,
        plots=True,
        save=True,
        val=True,
        amp=True,
    )
    best_pt = run_project / args.run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(best_pt)
    target = output_dir / "model_roll_aug_best.pt"
    shutil.copy2(best_pt, target)
    last_pt = run_project / args.run_name / "weights" / "last.pt"
    if last_pt.exists():
        shutil.copy2(last_pt, output_dir / "model_roll_aug_last.pt")
    return target


def export_onnx(best_pt: Path, output_dir: Path, imgsz: int) -> Path:
    model = YOLO(str(best_pt), task="classify")
    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=False)).resolve()
    target = output_dir / "model_roll_aug_best.onnx"
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    return target


def evaluate_best(
    best_pt: Path,
    original_dataset: Path,
    output_dir: Path,
    angles: list[int],
    args: argparse.Namespace,
) -> dict:
    model = YOLO(str(best_pt), task="classify")
    test_paths, test_labels, class_to_idx = collect_split(original_dataset, "test")
    full_test_metrics = evaluate_paths(model, test_paths, test_labels, class_to_idx, args.imgsz, args.batch, args.device)
    sampled_paths, sampled_labels = sample_by_class(test_paths, test_labels, args.roll_eval_per_class, args.seed)
    roll_metrics = {}
    for angle in angles:
        roll_metrics[str(angle)] = evaluate_paths(
            model,
            sampled_paths,
            sampled_labels,
            class_to_idx,
            args.imgsz,
            args.batch,
            args.device,
            angle=float(angle),
        )
    summary = {
        "best_pt": str(best_pt),
        "class_to_idx": class_to_idx,
        "test_images": len(test_paths),
        "roll_eval_images": len(sampled_paths),
        "full_test_metrics": full_test_metrics,
        "roll_angle_metrics": roll_metrics,
    }
    (output_dir / "roll_aug_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_against_original(
    roll_aug_best: Path,
    original_model: Path,
    original_dataset: Path,
    output_dir: Path,
    angles: list[int],
    args: argparse.Namespace,
) -> dict:
    test_paths, test_labels, class_to_idx = collect_split(original_dataset, "test")
    sampled_paths, sampled_labels = sample_by_class(test_paths, test_labels, args.roll_eval_per_class, args.seed)
    models = {
        "original": original_model,
        "roll_aug_finetuned": roll_aug_best,
    }
    comparison = {
        "sample_per_class": args.roll_eval_per_class,
        "sample_images": len(sampled_paths),
        "angles": angles,
        "models": {},
        "delta_roll_aug_minus_original": {},
    }
    for name, path in models.items():
        model = YOLO(str(path), task="classify")
        angle_metrics = {}
        for angle in angles:
            angle_metrics[str(angle)] = evaluate_paths(
                model,
                sampled_paths,
                sampled_labels,
                class_to_idx,
                args.imgsz,
                args.batch,
                args.device,
                angle=float(angle),
            )
        comparison["models"][name] = {
            "path": str(path),
            "roll_angle_metrics": angle_metrics,
        }

    for angle in angles:
        key = str(angle)
        comparison["delta_roll_aug_minus_original"][key] = {
            metric: comparison["models"]["roll_aug_finetuned"]["roll_angle_metrics"][key][metric]
            - comparison["models"]["original"]["roll_angle_metrics"][key][metric]
            for metric in ["accuracy", "close_precision", "close_recall", "f1", "pr_auc", "roc_auc"]
        }

    out = output_dir / "roll_aug_vs_original_metrics.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/yolov8n_cls_optuna/final_best_by_f1/model.pt")
    parser.add_argument("--dataset", default="_work/yolov8n_eye_state/extracted")
    parser.add_argument("--aug-dataset", default="_work/yolov8n_eye_state/roll_aug_data")
    parser.add_argument("--output", default="outputs/yolov8n_cls_optuna/05_roll_aug_dataset_finetune")
    parser.add_argument("--run-name", default="05_roll_aug_dataset_finetune_run")
    parser.add_argument("--angles", default="-50,-35,-20,0,20,35,50")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--build-workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=1.5e-4)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=3.5e-5)
    parser.add_argument("--translate", type=float, default=0.02)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roll-eval-per-class", type=int, default=1000)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    original_dataset = find_dataset_root(Path(args.dataset))
    aug_dataset = Path(args.aug_dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    angles = parse_angles(args.angles)

    config = vars(args).copy()
    config["angles"] = angles
    config["original_dataset"] = str(original_dataset)
    config["aug_dataset"] = str(aug_dataset)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not args.skip_build:
        manifest = build_roll_dataset(original_dataset, aug_dataset, angles, args.build_workers, args.force_build)
        print(json.dumps({"dataset_manifest": manifest}, indent=2))

    if args.build_only:
        print(f"Build-only complete: {aug_dataset}")
        return

    if args.skip_train:
        best_pt = output_dir / "model_roll_aug_best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"--skip-train requested but missing {best_pt}")
    else:
        best_pt = train_model(args, aug_dataset, output_dir)

    onnx_path = export_onnx(best_pt, output_dir, args.imgsz)
    metrics = evaluate_best(best_pt, original_dataset, output_dir, angles, args)
    comparison = evaluate_against_original(best_pt, Path(args.model).resolve(), original_dataset, output_dir, angles, args)
    final_summary = {
        "best_pt": str(best_pt),
        "onnx": str(onnx_path),
        "metrics": metrics,
        "comparison": comparison,
    }
    (output_dir / "roll_aug_final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
