import os
import sys
import subprocess
from pathlib import Path

# Auto-switch to project virtual environment if invoked by system Python
ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    print(f"[train.py] Re-launching with project virtual environment: {VENV_PYTHON}")
    res = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:])
    sys.exit(res.returncode)

import argparse
from ultralytics import YOLO


def find_latest_checkpoint(runs_dir: Path) -> Path | None:
    """Find the most recent last.pt weight file in runs directory."""
    checkpoints = sorted(runs_dir.glob("**/weights/last.pt"), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="Train or resume YOLO11 for Search & Rescue Robot")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "dataset_processed" / "data.yaml"),
        help="Path to data.yaml dataset definition",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Base model or checkpoint path (.pt)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4 for Windows)",
    )
    parser.add_argument("--patience", type=int, default=20, help="Patience epochs for early stopping")
    parser.add_argument(
        "--project",
        type=str,
        default=str(ROOT / "runs"),
        help="Directory to save run results",
    )
    parser.add_argument("--name", type=str, default="yolo11n_v2", help="Experiment name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest last.pt or specified checkpoint",
    )

    args = parser.parse_args()

    runs_dir = Path(args.project)
    if args.resume:
        checkpoint = args.weights
        if checkpoint == "yolo11n.pt":
            latest = find_latest_checkpoint(runs_dir)
            if latest:
                checkpoint = str(latest)
                print(f"[train.py] Found latest checkpoint to resume: {checkpoint}")
            else:
                sys.exit(f"[ERROR] No checkpoint (last.pt) found under {runs_dir} to resume from.")
        print(f"[train.py] Resuming training from: {checkpoint}")
        model = YOLO(checkpoint)
        model.train(resume=True)
    else:
        print(f"[train.py] Starting training with weights='{args.weights}', data='{args.data}', epochs={args.epochs}")
        model = YOLO(args.weights)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
        )


if __name__ == "__main__":
    main()