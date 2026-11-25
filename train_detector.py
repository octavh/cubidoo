"""Train the YOLO cube face detector on synthetic data.

    python train_detector.py                 # generate data (if missing) and train
    python train_detector.py --epochs 10     # quicker, less accurate

The best model is copied to models/cube_pose.pt, which main.py picks up automatically.
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from synth_data import generate
from yolo_detector import MODEL_PATH

ROOT = Path(__file__).parent


def pick_device():
    if torch.cuda.is_available():
        print(f"Training on NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return 0
    if torch.backends.mps.is_available():
        print("Training on Apple GPU (MPS)")
        return "mps"
    print("WARNING: no GPU found, training on CPU (this takes many hours). If this computer has an "
          "NVIDIA GPU, install the CUDA version of PyTorch - see README.")
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "datasets" / "cube_pose"))
    parser.add_argument("--train-images", type=int, default=6000)
    parser.add_argument("--val-images", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=str(ROOT / "models" / "yolo11n-pose.pt"),
                        help="pretrained starting point (downloaded automatically if missing)")
    parser.add_argument("--device", default=None, help="cpu, mps or a CUDA index (default: auto)")
    args = parser.parse_args()

    data_yaml = Path(args.data) / "data.yaml"
    if not data_yaml.exists():
        generate(args.data, args.train_images, args.val_images)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device if args.device is not None else pick_device(),
        degrees=0,       # rotation is already in the synthetic data
        fliplr=0.5,      # uses flip_idx from data.yaml to swap keypoint columns
        hsv_h=0.02,
        project=str(ROOT / "runs"),
        name="cube_pose",
        exist_ok=True,
    )

    best = ROOT / "runs" / "cube_pose" / "weights" / "best.pt"
    MODEL_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(best, MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
