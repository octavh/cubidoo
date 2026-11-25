# Cubidoo
Interactive Cube detector made with Computer Vision

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r libraries.txt
.venv/bin/python main.py
```

`main.py` uses the trained YOLO face detector when `models/cube_pose.pt` exists, otherwise the
classic contour-based detector. Add `--classic` to force the classic one (useful to compare).

## Trained face detector

A YOLO pose model finds each cube face and its 9 sticker centers. It is trained only on synthetic
images rendered from the virtual cube (random scrambles, poses, lighting, backgrounds, blur, glare,
fingers), so no manual labelling is needed.

```bash
.venv/bin/python synth_data.py --preview 12   # look at some training images (datasets/cube_pose/preview)
.venv/bin/python train_detector.py            # generate data + train, saves models/cube_pose.pt
```

Training takes about an hour on an Apple M-series GPU or an NVIDIA GPU, much longer on CPU only.
To make it work better in your room, put photos of your surroundings **without a cube** in
`backgrounds/`, delete `datasets/cube_pose` and train again.

### Training on another computer (e.g. Windows with an NVIDIA GPU)

```bat
git clone https://github.com/octavh/cubidoo.git
cd cubidoo
python -m venv .venv
.venv\Scripts\pip install -r libraries.txt
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
.venv\Scripts\python train_detector.py
git add models/cube_pose.pt
git commit -m "Add trained cube face model"
git push
```

The `torch` line installs the NVIDIA (CUDA) version of PyTorch; skip it on a Mac. If it fails,
get the right command for your CUDA version at https://pytorch.org/get-started/locally/. The script
prints which device it trains on - it should say "NVIDIA GPU", not "CPU". Everyone else then
gets the model with `git pull`.