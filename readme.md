# PACE-2025 Challenge - by Team EfficientAI

This repository contains the code and trained model for Team EfficientAI's winning submission to the PACE-2025 Challenge.

## Repository Structure

```
PACE_Inference/
├── checkpoints/
│   └── best_model.pth 
├── sample_input/
│   └── (Place your .png images here)
├── output/
│   └── (Results will be saved here)
├── src/
│   ├── main.py
│   ├── model.py
│   └── utils/
├── .gitattributes
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Setup and Installation

### 1. Prerequisites
- Python 3.8+
- Git and **Git LFS** (Large File Storage)

### 2. Install Git LFS
Our model file (`checkpoints/best_model.pth`)

- **On macOS (using Homebrew):**
  ```bash
  brew install git-lfs
  ```

After installing, run `git lfs install` once to initialize it.

### 3. Clone the Repository
Clone the repository and automatically download the large model file with Git LFS.

```bash
git clone https://github.com/Nimalesh/PACE_Inference.git
cd PACE_Inference
```
*(If you have already cloned the repository without Git LFS, run `git lfs pull` inside the directory to download the model file.)*

### 4. Create a Virtual Environment & Install Dependencies
This creates an isolated Python environment for the project.

```bash
# Create the environment
python3 -m venv venv
``` 
# Activate it
source venv/bin/activate

# Install required libraries
``` bash
pip install -r requirements.txt 

```

## Usage Instructions

The `src/main.py` script is the entry point for running inference. All commands should be run from the root `PACE_Inference` directory.

#### **Segmentation Task**
This will process all images in `./sample_input` and save the resulting binary masks in `./output/segmentation/`.

```bash
python src/main.py \
    --input ./sample_input \
    --output ./output \
    --task seg \
    --device cpu
```
#### **Classification Task**
This will process the same images and save a `predictions.csv` file in `./output/classification/`.

```bash
python src/main.py \
    --input ./sample_input \
    --output ./output \
    --task cls \
    --device cpu
```
*Note: If you have an Apple Silicon Mac, you can use `--device gpu` to leverage the MPS accelerator.*