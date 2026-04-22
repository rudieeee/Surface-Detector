# Surface Defect Detection - PPT Speaking Script + Teacher Questions

## How to use this document
- Read the Slide Script section while presenting.
- Replace bracket values like [XX] with your real results from your output graphs.
- Practice the Viva Q&A section before your presentation.

---

## Slide Script (What to Speak During PPT)

### Slide 1 - Title
"Good morning everyone. My project title is Surface Defect Detection using Convolutional Neural Networks with Grad-CAM explainability. The objective is to automatically classify steel surface defects into six classes and also explain why the model predicted a particular class."

### Slide 2 - Problem Statement
"In steel manufacturing, manual defect inspection is slow and can be inconsistent. So, this project solves a multi-class image classification problem where an input steel surface image is classified into one of six defect classes: Crazing, Inclusion, Patches, Pitted, Rolled, and Scratches."

### Slide 3 - Dataset and Folder Structure
"The dataset is organized class-wise into training and testing folders. The training images are in data/train_metal and testing images are in data/test_metal. This folder-based structure allows Keras flow_from_directory to automatically generate labels."

### Slide 4 - Technology Stack
"I used Python with TensorFlow/Keras for model building and training, scikit-learn for evaluation metrics like confusion matrix and ROC-AUC, matplotlib and seaborn for visualizations, and Pillow for image processing used in Grad-CAM overlay generation."

### Slide 5 - Data Preprocessing and Augmentation
"For training data, I applied rescaling, rotation, shifts, shear, zoom, horizontal and vertical flips, and brightness variation. This improves generalization and reduces overfitting. For validation data, only rescaling is applied so evaluation remains unbiased."

### Slide 6 - CNN Architecture
"The CNN has three convolution blocks with Batch Normalization and Max Pooling. After feature extraction, I used Flatten, Dropout at 0.5, a dense layer of 512 units, and a final softmax layer for six-class prediction. The model uses categorical cross-entropy loss and Adam optimizer with learning rate 1e-4."

### Slide 7 - Training Strategy (100 vs 300 vs 500)
"To answer the question of optimal training length, I compared three epoch budgets: 100, 300, and 500. I used EarlyStopping based on validation accuracy and ReduceLROnPlateau based on validation loss. This means even with higher max epochs, training stops automatically when performance saturates."

### Slide 8 - Best Setting Selection
"I recorded best validation accuracy, best validation loss, and actual epochs run for each setting. Then I selected the best setting based on highest validation accuracy and retrained the final model with that configuration."

Use your values while speaking:
- Best max-epoch setting: [XX]
- Best validation accuracy: [XX%]
- Best validation loss: [XX]
- Actual epochs run after early stopping: [XX]

### Slide 9 - Training Curves
"This graph shows train and validation accuracy and loss across epochs. If both curves improve and then stabilize with a small gap, it indicates healthy learning. A large train-validation gap would indicate overfitting."

### Slide 10 - Confusion Matrix and Class-wise Metrics
"The confusion matrix shows class-wise prediction performance. Diagonal values are correct predictions, and off-diagonal values indicate confusion between classes. I also computed precision, recall, and F1-score for each class to identify weak classes even if overall accuracy is high."

Use your values:
- Overall accuracy: [XX%]
- Balanced accuracy: [XX%]
- Weakest class by F1-score: [Class Name]
- Strongest class by F1-score: [Class Name]

### Slide 11 - ROC-AUC and Confidence Analysis
"I generated one-vs-rest ROC curves for all classes. AUC closer to 1 indicates better class separability. I also analyzed confidence distribution for correct vs wrong predictions to check if wrong predictions are overconfident."

### Slide 12 - Grad-CAM Explainability
"To make the model interpretable, I used Grad-CAM on the last convolution layer. For any test image, the heatmap highlights red/yellow regions the model focused on while making the prediction. This supports trust and helps verify that the model is attending to actual defect regions rather than background artifacts."

### Slide 13 - Limitations and Future Work
"Current limitations include possible class imbalance, domain shift in real factory conditions, and confusion between visually similar defects. Future work can include transfer learning, larger datasets, better calibration, and deployment as a real-time inspection tool."

### Slide 14 - Conclusion
"In conclusion, this project successfully performs multi-class steel defect classification and includes explainability through Grad-CAM. The epoch-budget comparison and detailed evaluation provide evidence-based model selection, and the approach can be extended for real industrial quality-control systems. Thank you."

---

## 1-Minute Project Summary (if teacher asks for short version)
"This project classifies steel surface defects into six classes using a CNN. I trained with data augmentation and compared three epoch budgets: 100, 300, and 500, with EarlyStopping and learning-rate reduction. I selected the best setup based on validation accuracy, then evaluated with confusion matrix, precision/recall/F1, ROC-AUC, and confidence analysis. Finally, I used Grad-CAM to show which image regions influenced predictions, improving model interpretability for practical quality-control use."

---

## Teacher Viva Questions (with sample answers)

### Basics and Motivation
1. Why did you choose this project?
Sample answer: "Surface defect detection is a real industrial problem where automation can reduce cost and human error. It is also a good computer vision use case with measurable outcomes."

2. Is this binary classification or multi-class classification?
Sample answer: "It is multi-class classification with six classes."

3. Why not use manual inspection only?
Sample answer: "Manual inspection is time-consuming, subjective, and hard to scale. A model can provide fast and consistent screening."

### Data and Preprocessing
4. Why did you use data augmentation?
Sample answer: "Augmentation increases data diversity and helps the model generalize by simulating variations like rotation, shifts, zoom, and lighting changes."

5. Why is augmentation not applied to validation data?
Sample answer: "Validation should represent unbiased performance. Applying strong augmentations in validation can distort true evaluation."

6. What input size did you use and why?
Sample answer: "I used 128x128 RGB images. It balances feature detail with computation cost."

7. How are labels assigned?
Sample answer: "Labels are inferred from class folder names using flow_from_directory, which creates one-hot encoded targets for categorical training."

### Model Design
8. Why CNN for this task?
Sample answer: "CNNs are effective for image feature extraction such as edges, textures, and local patterns that are important for defect recognition."

9. Why did you use Batch Normalization?
Sample answer: "It stabilizes and speeds up training by normalizing activations and reducing internal covariate shift."

10. Why Dropout(0.5)?
Sample answer: "Dropout is a regularization method that reduces overfitting by randomly deactivating neurons during training."

11. Why use softmax in output layer?
Sample answer: "Softmax converts logits into class probabilities that sum to 1, suitable for multi-class prediction."

12. Why categorical cross-entropy loss?
Sample answer: "Because labels are one-hot encoded and this is a multi-class classification problem."

### Training and Hyperparameters
13. Why compare 100, 300, and 500 epochs?
Sample answer: "To study how epoch budget influences performance and to justify model selection experimentally rather than by assumption."

14. If EarlyStopping is used, why still set 500 max epochs?
Sample answer: "A larger budget gives the model opportunity to improve if needed; EarlyStopping prevents unnecessary extra training."

15. What does ReduceLROnPlateau do?
Sample answer: "When validation loss plateaus, it reduces the learning rate so optimization can continue with finer updates."

16. Why monitor val_accuracy for EarlyStopping?
Sample answer: "Because final objective is better classification performance on unseen data, and validation accuracy directly reflects that."

### Evaluation and Interpretation
17. Difference between accuracy and balanced accuracy?
Sample answer: "Accuracy is total correct over total samples. Balanced accuracy averages recall across classes, so it is more reliable when classes are imbalanced."

18. What does confusion matrix tell you?
Sample answer: "It shows where the model is correct and which class pairs are confused, helping targeted improvements."

19. Why use precision, recall, and F1 together?
Sample answer: "Each captures different behavior. Precision handles false positives, recall handles false negatives, and F1 gives a harmonic balance."

20. What is ROC-AUC in multi-class setting?
Sample answer: "I used one-vs-rest ROC for each class. AUC measures how well each class is separated from others."

21. Can high accuracy still be misleading?
Sample answer: "Yes, especially with class imbalance. That is why class-wise metrics and balanced accuracy are important."

### Grad-CAM and Explainability
22. What is Grad-CAM in simple words?
Sample answer: "Grad-CAM creates a heatmap showing which image regions most influenced the model's decision for a chosen class."

23. Why is explainability needed here?
Sample answer: "In industrial use, users need confidence that predictions are based on defect regions, not background noise."

24. Which layer did you use for Grad-CAM and why?
Sample answer: "I used the last convolution layer because it captures high-level semantic features while preserving spatial information needed for localization."

### Practical and Improvement Questions
25. What are current limitations?
Sample answer: "Possible class imbalance, visual similarity between certain defects, and domain shift from lab data to real production environments."

26. How can you improve this model further?
Sample answer: "Use transfer learning, stronger data balancing, hyperparameter tuning, model calibration, and real-world data collection."

27. How would you deploy it in industry?
Sample answer: "Wrap the trained model in an API or edge application integrated with camera feed, then monitor drift and retrain periodically."

28. How do you handle wrong high-confidence predictions?
Sample answer: "Use confidence thresholding, calibration methods like temperature scaling, and human-in-the-loop review for low-trust cases."

29. How would you test robustness?
Sample answer: "Evaluate under lighting changes, blur/noise, camera angle variation, and unseen batches from new production days."

30. Why should we trust your result?
Sample answer: "Because I validated with multiple metrics, confusion analysis, and Grad-CAM interpretability instead of relying on a single accuracy number."

---

## Quick Fill Template (before final PPT)
- Best setting selected: [ ]
- Overall validation accuracy: [ ]
- Balanced accuracy: [ ]
- Best class by F1: [ ]
- Weak class by F1: [ ]
- Most common confusion pair: [ ]
- Mean per-class accuracy: [ ]
- One strong Grad-CAM example image: [ ]

---

## Final Tip for Presentation Delivery
- Speak in this order: Problem -> Method -> Results -> Explainability -> Limitations -> Future work.
- Keep numeric results ready on one cue card.
- For viva, always justify choices with reasoning, not only definitions.
