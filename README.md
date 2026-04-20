# 🌊 Reef AI Detection Suite

AI-powered detection and training tool for reef survey species — CoTS, Giant Clam, Bleaching, Fish and more. Runs entirely on your local machine with GPU support.

## Features
- YOLOv8 object detection with local GPU inference
- Manual annotation with Apple Pencil support on iPad
- Fine-tune and retrain models from your own survey images
- Image scraping from Bing, Google and Flickr for training data
- YouTube video frame extraction
- Training history dashboard
- Remote access via Cloudflare tunnel

## Requirements
- Windows 10/11
- Python 3.10+
- Node.js LTS
- NVIDIA GPU recommended (CPU works but is slow)
- Google Chrome (for browser capture feature)

## Setup (first time only)

1. Clone or download this repository
2. Install [Python 3.10+](https://python.org) and [Node.js LTS](https://nodejs.org)
3. Double-click `setup_dependencies.bat` — installs all Python and Node packages including CUDA PyTorch
4. Double-click `run_reef_ai.bat` — starts the app and opens your browser

## Updates

To get the latest version:
```
update.bat
```
Or manually: `git pull` then `setup_dependencies.bat` if new dependencies were added.

## Usage

1. **1. MODELS** — load a `.pt` model file (prefix with `cots_`, `fish_` etc to organise)
2. **2. UPLOAD** — add survey images, drop a folder, paste a YouTube URL, or scrape the web
3. **3. DETECT** — run AI detection on all images
4. **4. ANNOTATE** — correct mistakes, mark false positives, draw missed targets
5. **5. TRAIN** — export annotated images and fine-tune your model
6. **📊 HISTORY** — track model accuracy improvements over time
7. **🔍 SCRAPE** — download training images from the web by species name

## iPad / Remote Access

See the **REMOTE ACCESS** tab for Cloudflare tunnel setup instructions.

## Organisation
- `models/` — your trained `.pt` model files (not synced to GitHub)
- `datasets/` — extracted training datasets (not synced)
- `runs/` — YOLO training outputs (not synced)

## Built by
[Resilient Reefs Foundation](https://github.com/ResilientReefsFoundation)
