"""
Image Classification Project Using OpenCV and Scikit-learn
==========================================================
Train an image classification model using Machine Learning techniques
to recognize and classify images from the Fashion MNIST dataset.

Dataset: Fashion MNIST (70,000 grayscale images, 28x28 pixels, 10 classes)
Classes: T-shirt/top, Trouser, Pullover, Dress, Coat,
         Sandal, Shirt, Sneaker, Bag, Ankle boot

Pipeline:
  1. Load & explore the dataset
  2. Preprocess images with OpenCV
  3. Extract features (HOG + pixel-based)
  4. Train classifiers (SVM, Random Forest, KNN)
  5. Evaluate & compare models
  6. Visualize results & predictions
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import os
import gzip
import struct
import time
import urllib.request
import warnings

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
#  1. DATA LOADING & EXPLORATION
# ============================================================

BASE_URLS = [
    "https://github.com/zalandoresearch/fashion-mnist/raw/main/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images':  't10k-images-idx3-ubyte.gz',
    'test_labels':  't10k-labels-idx1-ubyte.gz',
}


def _download_file(filename, filepath):
    """Download a file with fallback URLs and clean error handling."""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return  # Already cached and non-empty

    import requests

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for base_url in BASE_URLS:
        url = base_url + filename
        print(f"    Downloading {filename} from {base_url}...")
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_path, filepath)
            print(f"    Successfully downloaded {filename}")
            return
        except Exception as e:
            print(f"    Failed from {base_url}: {e}")
            if os.path.exists(filepath + ".tmp"):
                try:
                    os.remove(filepath + ".tmp")
                except Exception:
                    pass

    raise RuntimeError(f"Failed to download {filename} from all available sources.")


def _parse_idx_images(filepath):
    """Parse IDX image file format."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols)


def _parse_idx_labels(filepath):
    """Parse IDX label file format."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


def load_data():
    """Load Fashion MNIST dataset from official source with local caching."""
    print("\n" + "=" * 60)
    print("  STEP 1: LOADING FASHION MNIST DATASET")
    print("=" * 60)

    print("  Downloading Fashion MNIST from official source...")
    print("  (Files are cached locally after first download)\n")

    # Download all files
    for key, filename in FILES.items():
        _download_file(filename, os.path.join(DATA_DIR, filename))

    # Parse files
    X_train = _parse_idx_images(os.path.join(DATA_DIR, FILES['train_images']))
    y_train = _parse_idx_labels(os.path.join(DATA_DIR, FILES['train_labels']))
    X_test = _parse_idx_images(os.path.join(DATA_DIR, FILES['test_images']))
    y_test = _parse_idx_labels(os.path.join(DATA_DIR, FILES['test_labels']))

    # Combine train + test
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    print(f"  ✅ Dataset loaded successfully!")
    print(f"  📊 Total samples   : {X.shape[0]:,}")
    print(f"  📐 Feature dims    : {X.shape[1]} (28×28 pixels)")
    print(f"  🏷️  Number of classes: {len(CLASS_NAMES)}")
    print(f"  📁 Classes: {', '.join(CLASS_NAMES)}")

    return X, y


def explore_data(X, y):
    """Visualize sample images from each class."""
    print("\n" + "=" * 60)
    print("  STEP 2: EXPLORING THE DATASET")
    print("=" * 60)

    # Display sample images from each class
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle('Fashion MNIST — Sample Images (One Per Class)',
                 fontsize=16, fontweight='bold', y=1.02)

    for i, ax in enumerate(axes.flat):
        idx = np.where(y == i)[0][0]
        img = X[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(CLASS_NAMES[i], fontsize=11, fontweight='600')
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sample_images.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Sample images saved to: {path}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(CLASS_NAMES)))
    bars = ax2.bar(CLASS_NAMES, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_xlabel('Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='600')
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'class_distribution.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✅ Class distribution saved to: {path2}")


# ============================================================
#  2. IMAGE PREPROCESSING WITH OPENCV
# ============================================================

def preprocess_image(image_flat):
    """
    Preprocess a single flattened image using OpenCV.
    - Reshape to 28x28
    - Apply Gaussian blur to reduce noise
    - Apply adaptive thresholding for better feature extraction
    - Resize to 32x32 for consistent feature extraction
    """
    img = image_flat.reshape(28, 28)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Resize to 32x32 for better HOG feature extraction
    resized = cv2.resize(blurred, (32, 32), interpolation=cv2.INTER_LINEAR)

    return resized


def extract_hog_features(image, cell_size=8, n_bins=9):
    """
    Extract HOG (Histogram of Oriented Gradients) features manually using OpenCV.
    Uses Sobel gradients instead of cv2.HOGDescriptor for maximum compatibility.
    """
    # Compute gradients using Sobel
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

    # Compute magnitude and angle
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180.0 / np.pi) % 180  # 0-180 degrees

    h, w = image.shape
    features = []

    # Compute histogram for each cell
    for i in range(0, h, cell_size):
        for j in range(0, w, cell_size):
            cell_mag = magnitude[i:i + cell_size, j:j + cell_size]
            cell_ang = angle[i:i + cell_size, j:j + cell_size]

            # Build histogram
            hist = np.zeros(n_bins)
            bin_width = 180.0 / n_bins
            for m, a in zip(cell_mag.ravel(), cell_ang.ravel()):
                bin_idx = int(a / bin_width) % n_bins
                hist[bin_idx] += m

            features.extend(hist)

    features = np.array(features, dtype=np.float64)

    # L2 normalization
    norm = np.linalg.norm(features) + 1e-6
    features = features / norm

    return features


def preprocess_and_extract_features(X, description=""):
    """
    Full preprocessing pipeline:
    1. Preprocess each image with OpenCV
    2. Extract HOG features
    3. Combine with normalized pixel features
    """
    print(f"\n  🔄 Processing {description} ({X.shape[0]:,} images)...")
    start = time.time()

    hog_features_list = []
    pixel_features_list = []

    total = X.shape[0]
    for i in range(total):
        # Preprocess
        processed_img = preprocess_image(X[i])

        # Extract HOG features
        hog_feat = extract_hog_features(processed_img)
        hog_features_list.append(hog_feat)

        # Normalized pixel features (flattened preprocessed image)
        pixel_feat = processed_img.flatten().astype(np.float64) / 255.0
        pixel_features_list.append(pixel_feat)

        # Progress indicator
        if (i + 1) % 5000 == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            elapsed = time.time() - start
            print(f"    [{pct:5.1f}%] Processed {i + 1:,}/{total:,} images "
                  f"({elapsed:.1f}s elapsed)")

    # Combine HOG + pixel features
    hog_array = np.array(hog_features_list)
    pixel_array = np.array(pixel_features_list)
    combined = np.hstack([hog_array, pixel_array])

    elapsed = time.time() - start
    print(f"  ✅ Feature extraction complete! ({elapsed:.1f}s)")
    print(f"     HOG features   : {hog_array.shape[1]}")
    print(f"     Pixel features : {pixel_array.shape[1]}")
    print(f"     Total features : {combined.shape[1]}")

    return combined


def visualize_preprocessing(X, y):
    """Show the preprocessing pipeline on sample images."""
    fig, axes = plt.subplots(3, 5, figsize=(14, 9))
    fig.suptitle('Image Preprocessing Pipeline', fontsize=16, fontweight='bold', y=1.02)

    row_labels = ['Original (28×28)', 'Blurred', 'Resized (32×32)']

    for col in range(5):
        idx = np.where(y == col * 2)[0][0]
        img = X[idx].reshape(28, 28)

        # Original
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(CLASS_NAMES[col * 2], fontsize=10, fontweight='600')
        axes[0, col].axis('off')

        # Blurred
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        axes[1, col].imshow(blurred, cmap='gray')
        axes[1, col].axis('off')

        # Resized
        resized = cv2.resize(blurred, (32, 32), interpolation=cv2.INTER_LINEAR)
        axes[2, col].imshow(resized, cmap='gray')
        axes[2, col].axis('off')

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='600', rotation=0,
                                 labelpad=100, va='center')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'preprocessing_pipeline.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Preprocessing visualization saved to: {path}")


# ============================================================
#  3. MODEL TRAINING
# ============================================================

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple classifiers."""
    print("\n" + "=" * 60)
    print("  STEP 4: TRAINING CLASSIFIERS")
    print("=" * 60)

    # Scale features
    print("\n  📏 Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'SVM (RBF Kernel)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=25, n_jobs=-1, random_state=42
        ),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  🏋️  Training: {name}")
        start = time.time()

        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start

        # Predictions
        start = time.time()
        y_pred = model.predict(X_test_scaled)
        pred_time = time.time() - start

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

        results[name] = {
            'model': model,
            'scaler': scaler,
            'accuracy': acc,
            'y_pred': y_pred,
            'train_time': train_time,
            'pred_time': pred_time,
            'report': report,
        }

        print(f"     ✅ Accuracy     : {acc:.4f} ({acc:.2%})")
        print(f"     ⏱️  Train time   : {train_time:.2f}s")
        print(f"     ⚡ Predict time : {pred_time:.2f}s")

    return results


# ============================================================
#  4. EVALUATION & VISUALIZATION
# ============================================================

def plot_model_comparison(results):
    """Bar chart comparing model accuracies."""
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in names]
    train_times = [results[n]['train_time'] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

    # Accuracy comparison
    colors = ['#6c63ff', '#10b981', '#f59e0b']
    bars1 = ax1.bar(names, accuracies, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')

    # Training time comparison
    bars2 = ax2.bar(names, train_times, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Training Time', fontsize=13, fontweight='bold')
    for bar, t in zip(bars2, train_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{t:.1f}s', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✅ Model comparison chart saved to: {path}")


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Confusion matrix saved to: {path}")


def visualize_predictions(X_test_raw, y_test, y_pred, model_name):
    """Show sample predictions with correct/incorrect labels."""
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(f'Sample Predictions — {model_name}',
                 fontsize=16, fontweight='bold', y=1.02)

    # 5 correct + 5 incorrect
    correct_idx = np.where(y_test == y_pred)[0]
    incorrect_idx = np.where(y_test != y_pred)[0]

    # Top row: correct predictions
    for i in range(5):
        idx = correct_idx[i]
        img = X_test_raw[idx].reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(
            f'✅ {CLASS_NAMES[y_pred[idx]]}',
            fontsize=10, color='green', fontweight='bold'
        )
        axes[0, i].axis('off')

    # Bottom row: incorrect predictions
    for i in range(min(5, len(incorrect_idx))):
        idx = incorrect_idx[i]
        img = X_test_raw[idx].reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(
            f'❌ {CLASS_NAMES[y_pred[idx]]}\n(True: {CLASS_NAMES[y_test[idx]]})',
            fontsize=9, color='red', fontweight='bold'
        )
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Correct', fontsize=12, fontweight='bold',
                           rotation=0, labelpad=60, va='center')
    axes[1, 0].set_ylabel('Wrong', fontsize=12, fontweight='bold',
                           rotation=0, labelpad=60, va='center')

    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    path = os.path.join(OUTPUT_DIR, f'predictions_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Prediction samples saved to: {path}")


def print_final_report(results):
    """Print the final summary report."""
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_acc = results[best_name]['accuracy']

    for name, res in results.items():
        marker = " 🏆 BEST" if name == best_name else ""
        print(f"\n  📌 {name}{marker}")
        print(f"     Accuracy     : {res['accuracy']:.4f} ({res['accuracy']:.2%})")
        print(f"     Train Time   : {res['train_time']:.2f}s")
        print(f"     Predict Time : {res['pred_time']:.2f}s")

    print(f"\n  🏆 Best Model: {best_name} ({best_acc:.2%})")
    print(f"\n  📂 All outputs saved to: {OUTPUT_DIR}/")

    # Print detailed classification report for best model
    print(f"\n  📋 Classification Report — {best_name}:")
    print("  " + "-" * 56)
    for line in results[best_name]['report'].split('\n'):
        print(f"  {line}")

    return best_name


# ============================================================
#  5. MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "🔸" * 30)
    print("  IMAGE CLASSIFICATION PROJECT")
    print("  Using OpenCV & Scikit-learn")
    print("  Dataset: Fashion MNIST")
    print("🔸" * 30)

    # Step 1: Load data
    X, y = load_data()

    # Step 2: Explore data
    explore_data(X, y)

    # Step 3: Preprocess & extract features
    print("\n" + "=" * 60)
    print("  STEP 3: PREPROCESSING & FEATURE EXTRACTION")
    print("=" * 60)

    visualize_preprocessing(X, y)

    # Use a subset for faster training (configurable)
    TRAIN_SIZE = 15000
    TEST_SIZE = 3000

    print(f"\n  📊 Using {TRAIN_SIZE:,} train + {TEST_SIZE:,} test samples")
    print(f"     (Adjust TRAIN_SIZE/TEST_SIZE in code for full dataset)\n")

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE,
        random_state=42, stratify=y
    )

    # Extract features
    X_train_feat = preprocess_and_extract_features(X_train_raw, "training set")
    X_test_feat = preprocess_and_extract_features(X_test_raw, "test set")

    # Step 4: Train models
    results = train_models(X_train_feat, X_test_feat, y_train, y_test)

    # Step 5: Evaluate & visualize
    print("\n" + "=" * 60)
    print("  STEP 5: EVALUATION & VISUALIZATION")
    print("=" * 60)

    plot_model_comparison(results)

    best_name = print_final_report(results)

    # Confusion matrix & predictions for the best model
    best_pred = results[best_name]['y_pred']
    plot_confusion_matrix(y_test, best_pred, best_name)
    visualize_predictions(X_test_raw, y_test, best_pred, best_name)

    for name, res in results.items():
        if name != best_name:
            plot_confusion_matrix(y_test, res['y_pred'], name)

    # Save the best model and scaler for real-world inference
    import joblib
    model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    joblib.dump(results[best_name]['model'], model_path)
    joblib.dump(results[best_name]['scaler'], scaler_path)
    print(f"\n  💾 Model exported to: {model_path}")
    print(f"  💾 Scaler exported to: {scaler_path}")

    print("\n" + "=" * 60)
    print("  ✅ PROJECT COMPLETE!")
    print(f"  📂 Check the 'output/' folder for all visualizations")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
