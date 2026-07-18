# 🔥 Fire Detection using CNN (MobileNetV2)

A deep learning model that detects fire in images using **Transfer Learning with MobileNetV2** — trained on real outdoor fire images with **98.87% accuracy**.

---

## Results

| Metric | Score |
|--------|-------|
| Training Accuracy | **98.87%** |
| Validation Accuracy | **98.49%** |
| Training Loss | 0.0417 |
| Validation Loss | 0.0743 |

---

## How It Works

The model takes an image as input and classifies it as either **FIRE** or **NO FIRE**.

```
Input Image (224x224)
        ↓
MobileNetV2 (pretrained on ImageNet) — Feature Extraction
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, ReLU) + Dropout(0.3)
        ↓
Dense(1, Sigmoid) → Fire Probability
```

Two-phase training strategy:
- **Phase 1** — MobileNetV2 base frozen, only classification head trained
- **Phase 2** — Last 30 layers unfrozen for fine-tuning at lower learning rate

---

## Dataset

**FIRE Dataset** by phylake1337 — originally created during NASA Space Apps Challenge 2018.

- 🔗 [kaggle.com/datasets/phylake1337/fire-dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- 755 outdoor fire images (with heavy smoke variants)
- 244 non-fire nature images (forest, river, lake, road, animals)
- Binary classification: `fire_images/` vs `non_fire_images/`

> Note: Dataset is slightly imbalanced (755 fire vs 244 non-fire). Data augmentation was used to compensate.

---

## Project Structure

```
Fire-Detection-CNN/
├── fire_detection_cnn.py     # Main training script
├── fire_detection_model.h5   # Saved trained model (after training)
├── training_curves.png       # Accuracy & loss plots (after training)
└── README.md
```

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- MobileNetV2 (ImageNet pretrained)
- NumPy
- Matplotlib
- KaggleHub

---

## Setup & Run

**1. Install dependencies**
```bash
pip install tensorflow matplotlib numpy kagglehub
```

**2. Download dataset**
```python
import kagglehub
path = kagglehub.dataset_download("phylake1337/fire-dataset")
print("Dataset path:", path)
```

**3. Update the path in the script**
```python
# In fire_detection_cnn.py, set this to your downloaded dataset path:
DATASET_DIR = r"C:\Users\<your_name>\.cache\kagglehub\datasets\phylake1337\fire-dataset\versions\1\fire_dataset"
```

**4. Run training**
```bash
python fire_detection_cnn.py
```

Training takes ~10-20 minutes on CPU, ~3-5 minutes on GPU.

---

## Predict on New Image

After training, use the built-in `predict_fire()` function:

```python
predict_fire(r"C:\path\to\your\image.jpg")

# Output:
# FIRE detected! (confidence: 97.3%)
# No fire. (confidence: 94.1%)
```

---

## Data Augmentation Used

| Technique | Value |
|-----------|-------|
| Rotation | ±20° |
| Zoom | 20% |
| Width / Height Shift | 10% |
| Horizontal Flip | Yes |
| Brightness Range | 0.7x – 1.3x |

---

## Author

**Mayank Singh**
- [LinkedIn](https://www.linkedin.com/in/mayank-singh-9b57a533a)
- [GitHub](https://github.com/mayank8030)

---

## License

This project is open source under the [MIT License](LICENSE).

> Dataset credits: phylake1337 on Kaggle — NASA Space Apps Challenge 2018.
