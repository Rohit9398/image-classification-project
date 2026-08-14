# 🖼️ Image Classification Using OpenCV & Scikit-learn

A machine learning project that trains image classifiers to recognize and classify **Fashion MNIST** images using **OpenCV** for image preprocessing and **Scikit-learn** for model training and evaluation.

---

## ✨ Features

- **Fashion MNIST Dataset** — 70,000 grayscale images across 10 clothing categories
- **OpenCV Preprocessing** — Gaussian blur, resizing, and noise reduction
- **HOG Feature Extraction** — Histogram of Oriented Gradients for shape-based recognition
- **3 ML Classifiers** — SVM (RBF), Random Forest, and KNN compared side-by-side
- **Comprehensive Evaluation** — Accuracy, precision, recall, F1-score, and confusion matrices
- **Auto-Generated Visualizations** — Sample images, preprocessing pipeline, model comparison, predictions

---

## 🏷️ Dataset — Fashion MNIST

| Class | Label |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Each image is **28×28 pixels**, grayscale. The dataset contains **60,000 training** and **10,000 test** images.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **OpenCV** | Image preprocessing (blur, resize, HOG) |
| **Scikit-learn** | ML models (SVM, Random Forest, KNN) |
| **Matplotlib** | Visualizations & charts |
| **NumPy** | Numerical computations |

---

## 📁 Project Structure

```
image-classification-project/
├── image_classifier.py    # Main script (preprocessing + training + evaluation)
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
└── output/                # Auto-generated visualizations
    ├── sample_images.png
    ├── class_distribution.png
    ├── preprocessing_pipeline.png
    ├── model_comparison.png
    ├── confusion_matrix_*.png
    └── predictions_*.png
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rohit9398/image-classification-project.git
   cd image-classification-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the classifier**
   ```bash
   python image_classifier.py
   ```

The script will automatically download the Fashion MNIST dataset, train 3 models, and save all visualizations to the `output/` folder.

---

## 🔬 How It Works

### Pipeline

```
Raw Images (28×28) → OpenCV Preprocessing → Feature Extraction (HOG + Pixels) → Model Training → Evaluation
```

### Step-by-Step

1. **Data Loading** — Downloads Fashion MNIST via `fetch_openml`
2. **Preprocessing** — Applies Gaussian blur and resizes to 32×32 using OpenCV
3. **Feature Extraction** — Extracts HOG descriptors (edge/gradient patterns) + normalized pixel values
4. **Training** — Trains SVM, Random Forest, and KNN classifiers on extracted features
5. **Evaluation** — Compares accuracy, generates confusion matrices, and visualizes predictions

---

## 📊 Models Compared

| Model | Description |
|---|---|
| **SVM (RBF Kernel)** | Support Vector Machine with radial basis function kernel — excellent for high-dimensional data |
| **Random Forest** | Ensemble of 200 decision trees — robust and handles noise well |
| **KNN (k=5)** | K-Nearest Neighbors — simple instance-based learning |

---

## 📈 Output Visualizations

The script auto-generates these visualizations in the `output/` folder:

| File | Description |
|---|---|
| `sample_images.png` | One sample image from each of the 10 classes |
| `class_distribution.png` | Bar chart showing number of samples per class |
| `preprocessing_pipeline.png` | Original → Blurred → Resized comparison |
| `model_comparison.png` | Accuracy and training time comparison across models |
| `confusion_matrix_*.png` | Confusion matrix for each classifier |
| `predictions_*.png` | Correct vs incorrect prediction samples |

---

## ⚙️ Configuration

You can adjust these variables in `image_classifier.py`:

```python
TRAIN_SIZE = 15000   # Number of training samples (max: ~56,000)
TEST_SIZE = 3000     # Number of test samples (max: ~14,000)
```

Increase for better accuracy (slower), decrease for faster execution.

---

## 📝 License

This project is open source and available for personal and educational use.

---

## 👤 Author

**Rohit Kumar**

- GitHub: [@Rohit9398](https://github.com/Rohit9398)

---

<p align="center">
  Built with ❤️ using Python • OpenCV • Scikit-learn • Matplotlib
</p>
