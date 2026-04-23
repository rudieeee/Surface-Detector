# ================================================================
#  SURFACE DEFECT DETECTION USING CNN + GRAD-CAM
# ================================================================

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import seaborn as sns
from PIL import Image


# ================================================================
# SECTION 1 — PATHS  
# ================================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, 'data', 'train_metal')
TESTING_DIR  = os.path.join(BASE_DIR, 'data', 'test_metal')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  SURFACE DEFECT DETECTION — STARTUP CHECK")
print("=" * 55)

if not os.path.exists(TRAINING_DIR):
    print(f"ERROR: Training folder not found at:\n  {TRAINING_DIR}")
    exit()

if not os.path.exists(TESTING_DIR):
    print(f"ERROR: Testing folder not found at:\n  {TESTING_DIR}")
    exit()

classes_found = sorted(os.listdir(TRAINING_DIR))
print(f"Classes found  : {classes_found}")
print(f"Training dir   : {TRAINING_DIR}")
print(f"Testing dir    : {TESTING_DIR}")
print(f"Output dir     : {OUTPUT_DIR}")
print("Startup check passed.\n")


# ================================================================
# SECTION 2 — SETTINGS
# ================================================================

IMG_WIDTH    = 128
IMG_HEIGHT   = 128
BATCH_SIZE   = 32

# ── Single epoch budget (early-stopping will find the true optimum) ──
EPOCH_VARIANTS = [300]


# ================================================================
# SECTION 3 — DATA GENERATORS
# ================================================================

print("--- Setting up Data Generators ---")

train_datagen = ImageDataGenerator(
    rescale          = 1./255,
    rotation_range   = 20,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    shear_range      = 0.2,
    zoom_range       = 0.2,
    horizontal_flip  = True,
    vertical_flip    = True,
    brightness_range = [0.8, 1.2],
    fill_mode        = 'nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical'
)

validation_generator = test_datagen.flow_from_directory(
    TESTING_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    shuffle     = False
)

NUM_CLASSES  = train_generator.num_classes
CLASS_NAMES  = list(train_generator.class_indices.keys())

print(f"Classes        : {CLASS_NAMES}")
print(f"Train samples  : {train_generator.samples}")
print(f"Val samples    : {validation_generator.samples}\n")


# ================================================================
# SECTION 4 — BUILD MODEL
# ================================================================

def build_model(num_classes, dropout_rate=0.5):
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4),
               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
       #Dropout(0.1),

        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
       # Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
       # Dropout(0.3),

        Flatten(),
       # Dropout(0.4),
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss      = 'categorical_crossentropy',
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics   = ['accuracy']
    )
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1)
    ]


# ================================================================
# SECTION 5 — TRAIN
# ================================================================

print("=" * 55)
print("  TRAINING — max_epochs = 300  (early-stopping active)")
print("=" * 55)

histories       = {}
results_summary = {}

for MAX_EPOCHS in EPOCH_VARIANTS:
    print(f"\n>>> Training with max_epochs = {MAX_EPOCHS}")
    m = build_model(NUM_CLASSES)
    h = m.fit(
        train_generator,
        epochs          = MAX_EPOCHS,
        validation_data = validation_generator,
        callbacks       = get_callbacks(),
        verbose         = 1
    )
    histories[MAX_EPOCHS] = h
    results_summary[MAX_EPOCHS] = {
        'val_accuracy'  : max(h.history['val_accuracy']),
        'val_loss'      : min(h.history['val_loss']),
        'actual_epochs' : len(h.history['accuracy'])
    }
    print(f"    Best val_accuracy: {results_summary[MAX_EPOCHS]['val_accuracy']:.4f}  "
          f"(stopped at epoch {results_summary[MAX_EPOCHS]['actual_epochs']})")

best_setting = max(results_summary, key=lambda k: results_summary[k]['val_accuracy'])
print(f"\n>>> Best setting: max_epochs={best_setting}  "
      f"acc={results_summary[best_setting]['val_accuracy']:.4f}")

print(f"\n>>> Re-training final model with max_epochs={best_setting} ...")
model = build_model(NUM_CLASSES)
final_history = model.fit(
    train_generator,
    epochs          = best_setting,
    validation_data = validation_generator,
    callbacks       = get_callbacks(),
    verbose         = 1
)

# Warm up model so symbolic tensors are defined (needed for Grad-CAM)
_ = model.predict(np.zeros((1, IMG_WIDTH, IMG_HEIGHT, 3)), verbose=0)

# Print all layer names so you always know what's available
print("\n--- Model Layer Names ---")
for i, layer in enumerate(model.layers):
    print(f"  [{i:2d}] {layer.name:40s}  type: {type(layer).__name__}")


# ================================================================
# SECTION 6 — GRAPH 1+2: Training curves
# ================================================================

print("\n--- Saving Graph 1+2: Training Curves ---")

h = final_history.history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training Curves — Final Model', fontsize=14, fontweight='bold')

axes[0].plot(h['accuracy'],     label='Train Accuracy', linewidth=2)
axes[0].plot(h['val_accuracy'], label='Val Accuracy',   linewidth=2)
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy'); axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.4); axes[0].legend()

axes[1].plot(h['loss'],     label='Train Loss', linewidth=2)
axes[1].plot(h['val_loss'], label='Val Loss',   linewidth=2)
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss'); axes[1].grid(True, alpha=0.4); axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_training_curves.png")


# ================================================================
# SECTION 7 — GRAPH 2: Training summary (single run)
# ================================================================

print("\n--- Saving Graph 2: Training Summary ---")

r = results_summary[300]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(
    'Training Summary — 300-epoch Budget\n'
    '(Early-stopping finds the true optimal point automatically)',
    fontsize=13, fontweight='bold'
)

axes[0].bar(['300'], [r['val_accuracy']], color='#5B8DB8', edgecolor='black', linewidth=0.6)
axes[0].set_title('Best Validation Accuracy')
axes[0].set_xlabel('Max Epoch Budget'); axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1.1])
axes[0].text(0, r['val_accuracy'] + 0.02, f"{r['val_accuracy']:.3f}",
             ha='center', fontweight='bold')

axes[1].bar(['300'], [r['actual_epochs']], color='#4CAF50', edgecolor='black', linewidth=0.6)
axes[1].set_title('Actual Epochs Run\n(early-stopping effect)')
axes[1].set_xlabel('Max Epoch Budget'); axes[1].set_ylabel('Epochs run')
axes[1].text(0, r['actual_epochs'] + 1, str(r['actual_epochs']),
             ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_training_summary.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_training_summary.png")

print("\n" + "=" * 60)
print("   TRAINING SUMMARY")
print("=" * 60)
print(f"  Max Epoch Budget : 300")
print(f"  Actual Epochs Run: {r['actual_epochs']}")
print(f"  Best Val Accuracy: {r['val_accuracy']:.4f}")
print(f"  Best Val Loss    : {r['val_loss']:.4f}")
print("=" * 60)


# ================================================================
# SECTION 8 — EVALUATION GRAPHS
# ================================================================

print("\n--- Running Evaluation ---")

validation_generator.reset()
y_pred_probs   = model.predict(validation_generator, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true         = validation_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 03_confusion_matrix.png")

report_dict = classification_report(y_true, y_pred_classes,
                                     target_names=CLASS_NAMES, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
print("\n--- Detailed Metrics ---")
print(df_report.round(2))

f1_scores = [report_dict[c]['f1-score']  for c in CLASS_NAMES]
precision  = [report_dict[c]['precision'] for c in CLASS_NAMES]
recall     = [report_dict[c]['recall']    for c in CLASS_NAMES]

x = np.arange(len(CLASS_NAMES)); width = 0.25
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width, precision, width, label='Precision', color='#5B8DB8')
ax.bar(x,         f1_scores, width, label='F1 Score',  color='#E07B54')
ax.bar(x + width, recall,    width, label='Recall',    color='#4CAF50')
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylim(0, 1.2); ax.set_ylabel('Score')
ax.set_title('Precision / F1 / Recall per Class', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_precision_f1_recall.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_precision_f1_recall.png")

y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
fig, ax    = plt.subplots(figsize=(8, 6))
for i, cname in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f'{cname} (AUC={roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Multi-class', fontsize=13, fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_roc_curves.png")

max_confs = np.max(y_pred_probs, axis=1)
correct   = (y_pred_classes == y_true)
fig, ax   = plt.subplots(figsize=(8, 4))
ax.hist(max_confs[correct],  bins=20, alpha=0.7, color='#4CAF50', label='Correct')
ax.hist(max_confs[~correct], bins=20, alpha=0.7, color='#E07B54', label='Wrong')
ax.set_xlabel('Prediction Confidence'); ax.set_ylabel('Count')
ax.set_title('Confidence Distribution: Correct vs Wrong Predictions', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_confidence_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_confidence_distribution.png")

per_class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = ['#4CAF50' if v >= 0.9 else '#E07B54' for v in per_class_acc]
bars = ax.bar(CLASS_NAMES, per_class_acc, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.axhline(np.mean(per_class_acc), color='navy', linestyle='--',
           linewidth=1.5, label=f'Mean = {np.mean(per_class_acc):.2f}')
ax.set_ylim(0, 1.15); ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
for bar, v in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
            f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_per_class_accuracy.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_per_class_accuracy.png")


# ================================================================
# SECTION 9 — GRAD-CAM FUNCTIONS  (FIXED for Keras 3)
# ================================================================

def get_last_conv_layer_name(model):
    """
    Auto-detects the last Conv2D layer name.
    NEVER hardcode 'conv2d_2' — it changes every time you retrain.
    """
    last_conv_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
    if last_conv_name is None:
        raise ValueError("No Conv2D layer found in model.")
    print(f"[Grad-CAM] Auto-detected last Conv2D layer: '{last_conv_name}'")
    return last_conv_name


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Keras 3 compatible Grad-CAM.

    Instead of building a sub-Model from model.inputs / model.outputs
    (which fails on Sequential models that haven't been traced as a
    functional graph), we do a manual layer-by-layer forward pass
    inside GradientTape, watching the conv output directly.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        x        = img_tensor
        conv_out = None

        # Walk every layer; record output of the target conv layer
        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                conv_out = x
                tape.watch(conv_out)   # ← watch the conv tensor, not the input

        preds = x   # final softmax output

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads        = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)   # ReLU

    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape)      # blank if no activation
    heatmap = heatmap / max_val             # normalize 0 → 1
    return heatmap.numpy()


def get_img_and_heatmap(img_display, img_array, model, last_conv_layer_name):
    """
    Returns: (superimposed PIL image, predicted class index, confidence)
    img_display : original numpy uint8  (H × W × 3)
    img_array   : preprocessed float32  (1 × 128 × 128 × 3)
    """
    preds      = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

    # Colorize heatmap with jet colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet           = plt.cm.jet(np.arange(256))[:, :3]       # (256, 3) float 0-1
    jet_heatmap   = jet[heatmap_uint8]                       # (H', W', 3)
    jet_heatmap   = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(
                        (img_display.shape[1], img_display.shape[0]),  # (W, H)
                        Image.BILINEAR
                    )
    jet_heatmap = np.array(jet_heatmap).astype(float)

    # Blend: 40% heatmap + 60% original
    orig_float   = img_display.astype(float)
    superimposed = np.clip(jet_heatmap * 0.4 + orig_float * 0.6, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed), pred_index, confidence


# ================================================================
# SECTION 10 — GRAD-CAM PREDICTION ON YOUR OWN IMAGE
# ================================================================

def predict_image_with_gradcam(model, img_width, img_height, class_names):

    # ── Auto-detect last conv layer — never hardcode ──────────
    LAST_CONV_LAYER = get_last_conv_layer_name(model)

    print("\n--- Grad-CAM Prediction ---")
    print("Enter the full path to a steel surface image on your PC.")
    print("Example: /Users/yourname/Desktop/steel_sample.jpg")
    test_image_path = input("\nImage path: ").strip().strip('"')

    if not os.path.exists(test_image_path):
        print(f"ERROR: File not found at {test_image_path}")
        return

    fn          = os.path.basename(test_image_path)
    img         = Image.open(test_image_path).convert('RGB')
    img_display = np.array(img)                              # original size, uint8
    img_resized = img.resize((img_width, img_height))
    img_array   = np.expand_dims(
                      np.array(img_resized) / 255.0, axis=0
                  ).astype(np.float32)

    print(f"Image loaded   : {img_display.shape}")
    print(f"Model input    : {img_array.shape}")
    print(f"Grad-CAM layer : {LAST_CONV_LAYER}")

    gradcam_img, pred_index, confidence = get_img_and_heatmap(
        img_display, img_array, model, LAST_CONV_LAYER
    )
    predicted_label = class_names[pred_index]

    # ── Side-by-side: original + heatmap ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_display)
    axes[0].set_title(f'Original Image: {fn}', fontsize=11)
    axes[0].axis('off')
    axes[1].imshow(gradcam_img)
    axes[1].set_title(f'Grad-CAM: {predicted_label} ({confidence*100:.2f}%)', fontsize=11)
    axes[1].axis('off')
    plt.suptitle('Red/Yellow areas = where the model detected the defect',
                 fontsize=9, color='gray')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'gradcam_{fn}')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")

    # ── Top-3 confidence bar chart ─────────────────────────────
    all_preds = model.predict(img_array, verbose=0)[0]
    top3_idx  = np.argsort(all_preds)[::-1][:3]
    top3_lbls = [class_names[i] for i in top3_idx]
    top3_vals = [all_preds[i] * 100 for i in top3_idx]

    fig, ax = plt.subplots(figsize=(6, 3))
    bar_colors = ['#4CAF50' if i == 0 else '#9E9E9E' for i in range(3)]
    bars = ax.barh(top3_lbls[::-1], top3_vals[::-1], color=bar_colors[::-1])
    for bar, v in zip(bars, top3_vals[::-1]):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=10)
    ax.set_xlim(0, 115); ax.set_xlabel('Confidence (%)')
    ax.set_title('Top-3 Predictions', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out_path2 = os.path.join(OUTPUT_DIR, f'top3_{fn}')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path2}")

    print(f"\n--- Prediction Result (Grad-CAM Analysis) ---")
    print(f"Model Prediction : {predicted_label}")
    print(f"Confidence       : {confidence*100:.2f}%")
    print(f"Grad-CAM layer   : {LAST_CONV_LAYER}")
    print("Red/yellow areas in the heatmap = where the model detected the defect.")


# Run prediction
predict_image_with_gradcam(model, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)

print("\n\nAll done! Check the 'outputs/' folder for all saved graphs.")