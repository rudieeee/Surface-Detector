# Surface Defect Detection Project Documentation

## 1) Project Overview
This project detects steel surface defects using a Convolutional Neural Network (CNN) trained on image folders.

It performs:
- Multi-class defect classification
- Epoch-budget comparison (100 vs 300 vs 500)
- Detailed evaluation with multiple plots
- Grad-CAM explainability on custom images

Defect classes used:
- Crazing
- Inclusion
- Patches
- Pitted
- Rolled
- Scratches

---

## 2) Problem Statement
Given an input image of a steel surface, predict which defect class it belongs to.

This is a supervised multi-class image classification problem.

---

## 3) Project Structure
Main script:
- main.py

Key folders:
- data/train_metal: Training images (class-wise subfolders)
- data/test_metal: Validation/test images (class-wise subfolders)
- outputs: All generated plots and Grad-CAM outputs
- training_data/annotations and testing_data/annotations: XML annotation files (available in project)

---

## 4) End-to-End Pipeline
1. Validate dataset paths.
2. Load images with ImageDataGenerator.
3. Apply augmentation for training data.
4. Build CNN model.
5. Train 3 experiments with max epochs: 100, 300, 500.
6. Use EarlyStopping and ReduceLROnPlateau callbacks.
7. Select best epoch budget based on best validation accuracy.
8. Re-train final model with best epoch budget.
9. Generate evaluation plots and reports.
10. Run Grad-CAM for user-selected custom image.

---

## 5) Technologies Used
Core stack:
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

What each library does:
- TensorFlow/Keras: CNN model, training loop, callbacks
- Scikit-learn: confusion matrix, classification report, ROC, AUC
- Matplotlib/Seaborn: all graphs and heatmaps
- NumPy/Pandas: numerical operations and metrics table
- Pillow: image loading and Grad-CAM overlays

---

## 6) Model Architecture
The CNN in main.py uses 3 convolution blocks plus a dense classifier.

Architecture summary:
- Conv2D(32, 3x3, relu, same) + BatchNorm + MaxPool
- Conv2D(64, 3x3, relu, same) + BatchNorm + MaxPool
- Conv2D(128, 3x3, relu, same) + BatchNorm + MaxPool
- Flatten
- Dropout(0.5)
- Dense(512, relu)
- Dense(num_classes, softmax)

Compilation:
- Loss: categorical_crossentropy
- Optimizer: Adam(learning_rate = 1e-4)
- Metric: accuracy

---

## 7) Data Preparation and Augmentation
Training augmentation used:
- Rescale by 1/255
- Rotation
- Width shift
- Height shift
- Shear
- Zoom
- Horizontal flip
- Vertical flip
- Brightness range changes

Validation data:
- Only rescale by 1/255
- No augmentation

Why this matters:
- Augmentation improves generalization
- Validation without augmentation gives a cleaner performance estimate

---

## 8) Training Strategy
Epoch budgets tested:
- 100
- 300
- 500

Callbacks used:
- EarlyStopping on validation accuracy with patience 20
- ReduceLROnPlateau on validation loss with factor 0.5 and patience 8

Important behavior:
- EarlyStopping may stop training earlier than max epochs
- Higher max epoch budget can still help because model gets more opportunity before early stop

---

## 9) How Accuracy Is Calculated
There are multiple accuracy views in this project.

### 9.1 Training/Validation Accuracy (Keras categorical accuracy)
For each sample:
- Predicted class = argmax of softmax probabilities
- True class = one-hot label index
- Correct sample if predicted class equals true class

Accuracy formula:
Accuracy = (Number of correct predictions) / (Total predictions)

If N samples are evaluated and c are correct:
Accuracy = c / N

### 9.2 Overall Accuracy from Confusion Matrix
Let confusion matrix be M where M(i, j) is count of true class i predicted as class j.

Overall accuracy:
Accuracy = sum of diagonal entries / sum of all entries

Equivalent meaning:
- Diagonal = correct predictions
- All cells = total predictions

### 9.3 Per-Class Accuracy
For each class i:
Per-class accuracy(i) = M(i, i) / sum of row i

Meaning:
- Out of all true samples of class i, how many were predicted correctly

---

## 10) Other Metrics in This Project
Precision for class i:
Precision(i) = TP(i) / (TP(i) + FP(i))

Recall for class i:
Recall(i) = TP(i) / (TP(i) + FN(i))

F1-score for class i:
F1(i) = 2 * Precision(i) * Recall(i) / (Precision(i) + Recall(i))

ROC and AUC:
- One-vs-rest ROC curve is computed for each class
- AUC closer to 1 means better separation capability

---

## 11) Graph-by-Graph Explanation
All main plots are saved in outputs.

### Graph 1 and 2: Training Curves
File:
- outputs/01_training_curves.png

What it shows:
- Train and validation accuracy vs epoch
- Train and validation loss vs epoch

How to interpret:
- Large train-val gap can indicate overfitting
- Both curves improving and stabilizing suggests healthy training

### Graph 3: Epoch Comparison
File:
- outputs/02_epoch_comparison.png

What it shows:
- Best validation accuracy for each max epoch budget
- Best validation loss for each max epoch budget
- Actual epochs run after early stopping

How to interpret:
- Compares effect of epoch budget under early stopping
- Helps justify final chosen budget

### Graph 4: Confusion Matrix
File:
- outputs/03_confusion_matrix.png

What it shows:
- True classes on y-axis, predicted on x-axis
- Diagonal cells are correct predictions

How to interpret:
- High diagonal values are good
- Off-diagonal highlights class confusions

### Graph 5: Precision/F1/Recall per Class
File:
- outputs/04_precision_f1_recall.png

What it shows:
- Per-class bars for precision, recall, and F1-score

How to interpret:
- Finds weak classes even if overall accuracy is high

### Graph 6: ROC Curves
File:
- outputs/05_roc_curves.png

What it shows:
- One-vs-rest ROC curve per class
- AUC value in legend per class

How to interpret:
- Curves near top-left and high AUC are better

### Graph 7: Confidence Distribution
File:
- outputs/06_confidence_distribution.png

What it shows:
- Histogram of max softmax confidence for correct vs wrong predictions

How to interpret:
- If wrong predictions have high confidence, model calibration may be weak

### Graph 8: Per-Class Accuracy
File:
- outputs/07_per_class_accuracy.png

What it shows:
- Accuracy per class from confusion matrix rows
- Mean per-class accuracy line

How to interpret:
- Quickly identifies hardest classes

### Grad-CAM Outputs on Custom Image
Files generated per input image:
- outputs/gradcam_<filename>
- outputs/top3_<filename>

What they show:
- Heatmap overlay highlighting regions influencing prediction
- Top-3 predicted classes with confidence percentages

---

## 12) Ways to Change and Improve Accuracy
You can increase or control accuracy using these methods.

### Data improvements
- Add more labeled images for weak classes
- Balance class counts to reduce class bias
- Clean mislabeled or low-quality samples
- Keep train/test split leakage-free

### Augmentation tuning
- Increase augmentation if overfitting
- Reduce aggressive augmentation if underfitting
- Add contrast or noise augmentation for robustness

### Model improvements
- Increase image size (for finer defect details)
- Add one more conv block or use GlobalAveragePooling
- Tune dropout (lower if underfitting, higher if overfitting)
- Use transfer learning backbones (for example EfficientNet, ResNet)

### Optimization and training
- Tune learning rate (often most impactful)
- Try different batch sizes (smaller can regularize, larger can stabilize)
- Train longer with better schedule (cosine decay, warmup)
- Keep early stopping but tune patience

### Class imbalance handling
- Use class weights in model.fit
- Use focal loss for hard/rare classes
- Oversample minority classes

### Decision quality and reliability
- Add confidence threshold to reject uncertain predictions
- Calibrate probabilities (temperature scaling)
- Use test-time augmentation for more stable prediction

### Evaluation discipline
- Use per-class metrics, not only global accuracy
- Track confusion matrix changes after each tuning step
- Compare with cross-validation if dataset size is limited

Practical rule:
- If train accuracy high and val accuracy low: reduce overfitting (more data, stronger regularization, augmentation).
- If both are low: increase model capacity, improve features/data quality, or tune optimizer/lr.

---

## 13) How to Run
From project root:
- python3 main.py

After training/evaluation, check generated outputs in:
- outputs

---

## 14) Suggested Next Upgrades
- Save best model to disk and load for inference
- Add command-line arguments for epochs, learning rate, and image size
- Export metrics to CSV for experiment tracking
- Add TensorBoard logging
- Add a small web UI for drag-and-drop prediction with Grad-CAM

---

## 15) Quick Summary
This project is a full deep-learning defect classification pipeline with:
- Strong augmentation
- CNN training and epoch-budget experiments
- Rich evaluation plots
- Explainability with Grad-CAM

It is already structured well for coursework and can be improved further by transfer learning, class balancing, and systematic hyperparameter tuning.
