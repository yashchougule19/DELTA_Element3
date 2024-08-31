from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationEvaluator:
    def __init__(self, true_labels, predicted_labels, predicted_probs=None):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.predicted_probs = predicted_probs

    def evaluate(self):
        print('Classification report:\n')
        print(classification_report(self.true_labels, self.predicted_labels))

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        plt.figure(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative (0)', 'Positive (1)'], 
                    yticklabels=['Negative (0)', 'Positive (1)'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_auc(self):
        fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        
        print(f'AUC: {roc_auc:.2f}')
        
    def plot_precision_recall_auc(self):
        
        # Flip the true labels to ensure precision and recall are calculated with respect to the minority class
        flipped_labels = [1 if label == 0 else 0 for label in self.true_labels]
        
        # Compute precision, recall and auc with flipped labels
        precision, recall, _ = precision_recall_curve(flipped_labels, self.predicted_probs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(4.5, 3))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})', color='blue')
        plt.xlabel('True Positive Rate (Recall)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for minority class (0)')
        plt.legend(loc='upper right')
        plt.show()