from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        
        print(f'AUC: {roc_auc:.2f}')