# Sign Language to Text Conversion System

A real-time American Sign Language (ASL) recognition system that converts hand gestures captured via webcam into text using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Features

- **Real-time webcam gesture detection** with OpenCV handoff background removal
- **CNN-based classification** of ASL alphabet (A–Z) and common words
- **Sign-to-text conversion** with continuous sentence formation
- **Prediction confidence** display and gesture smoothing
- **Full ML pipeline**: dataset prep, augmentation, train/val/test split, training, evaluation
- **Scientific optimization**: configurable conv layers, pooling, dropout, batch norm, optimizers, learning rates
- **Evaluation dashboard**: accuracy, precision, recall, F1, confusion matrix, loss curves
- **Logging and performance monitoring** (FPS, latency)

## Project Structure

```
Sign Language/
├── app/main.py              # Real-time recognition GUI
├── config/config.yaml       # System configuration
├── scripts/
│   ├── prepare_dataset.py   # Dataset preparation
│   ├── train_model.py       # Model training
│   ├── evaluate_model.py    # Model evaluation
│   └── run_experiments.py   # Architecture experiments
├── src/
│   ├── preprocessing/       # Hand detection, augmentation, dataset
│   ├── models/              # CNN architecture & experiments
│   ├── training/            # Trainer & evaluator
│   ├── inference/           # Real-time recognizer
│   └── utils/               # Config, logging, performance
├── data/
│   ├── raw/                 # Raw images (folder per class)
│   ├── processed/           # Preprocessed numpy arrays
│   └── models/              # Saved models & training runs
├── outputs/                 # Evaluation plots & reports
└── logs/                    # Application logs
```

## Quick Start

### 1. Install Dependencies

**Requires Python 3.10, 3.11, or 3.12** (TensorFlow is not available for Python 3.13+).

```bash
cd "d:\Projects\Sign Language"
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_environment.py
```

### 2. Prepare Dataset (real ASL images required)

Download [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), extract the zip, then:

```bash
python scripts/import_asl_alphabet.py --source "PATH_TO/asl_alphabet_train/asl_alphabet_train"
python scripts/prepare_dataset.py
```

### 3. Train the Model

```bash
python scripts/train_model.py --epochs 25
```

Uses **MobileNetV2** transfer learning on 128×128 images (A–Z only).

Training saves the best model to `data/models/run_*/best_model.keras` and evaluation artifacts to `outputs/evaluation/`.

### 4. Run Real-Time Recognition

```bash
python app/main.py
```

**Controls:**
| Key | Action |
|-----|--------|
| Q | Quit |
| C | Clear sentence |
| B | Backspace last sign |
| Space | Add space to sentence |

## Model Architecture

Default CNN (`asl_cnn_v1`):

| Layer | Configuration |
|-------|---------------|
| Conv Block 1 | 32 filters, 3×3, ReLU, BatchNorm, MaxPool |
| Conv Block 2 | 64 filters, 3×3, ReLU, BatchNorm, MaxPool |
| Conv Block 3 | 128 filters, 3×3, ReLU, BatchNorm, MaxPool |
| Conv Block 4 | 256 filters, 3×3, ReLU, BatchNorm, MaxPool |
| Dense | 512 → 256 units, BatchNorm, Dropout (0.5) |
| Output | Softmax (41 classes) |

## Experimentation

Run predefined architecture experiments:

```bash
python scripts/run_experiments.py
python scripts/train_model.py --experiment deep_4conv
```

Experiments vary: conv depth, kernel sizes, activations (ReLU vs LeakyReLU), dropout rates, and optimizers (Adam, SGD, RMSprop).

## Evaluation Metrics

After training, find results in `outputs/evaluation/`:

- `evaluation_report.json` — accuracy, precision, recall, F1
- `confusion_matrix.png` — per-class confusion matrix
- `loss_curves.png` — training/validation loss and accuracy

Evaluate a specific model:

```bash
python scripts/evaluate_model.py --model data/models/run_*/best_model.keras
```

## Configuration

Edit `config/config.yaml` to adjust:

- Image size, augmentation parameters
- CNN architecture (filters, layers, dropout)
- Training hyperparameters (epochs, learning rate, optimizer)
- Inference thresholds and sentence building
- Webcam settings

## Git & GitHub

```bash
git init
git add .
git commit -m "Initial commit: ASL sign language recognition system"
git remote add origin https://github.com/YOUR_USERNAME/sign-language-recognition.git
git push -u origin main
```

## Performance Notes

- **98%+ training accuracy** is achievable with the Kaggle ASL Alphabet dataset and MobileNetV2.
- If signs are wrong, you are likely still using an old model trained on synthetic data — re-import and retrain.
- Hold each letter sign steady for ~1 second; keep hand below your face with plain background.

## Tech Stack

Python · TensorFlow · Keras · OpenCV · NumPy · Matplotlib · Scikit-learn · Git

## License

MIT
