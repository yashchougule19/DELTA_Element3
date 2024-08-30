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
        
        
        


class ClassificationEvaluator:
    def __init__(self, true_labels, predicted_labels=None, predicted_probs=None, cutoff=0.5):
        self.true_labels = true_labels
        self.predicted_probs = predicted_probs
        self.cutoff = cutoff
        
        if predicted_probs is not None:
            self.predicted_labels = (np.array(predicted_probs) >= cutoff).astype(int)
        else:
            self.predicted_labels = predicted_labels

    def evaluate(self):
        if self.predicted_probs is not None:
            print(f'Using cutoff of {self.cutoff} to binarize predicted probabilities.')
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
        if self.predicted_probs is None:
            raise ValueError("Predicted probabilities are required to plot ROC AUC.")
        
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

    def assess_model(self, plot_roc=True):
        """
        Evaluate the model, calculate accuracy, AUC, and optionally plot the ROC curve.
        """
        if self.predicted_probs is not None:
            yhat_c = (np.array(self.predicted_probs) >= self.cutoff).astype(int)
        else:
            yhat_c = self.predicted_labels
        
        acc = accuracy_score(self.true_labels, yhat_c)
        auc_value = roc_auc_score(self.true_labels, self.predicted_probs) if self.predicted_probs is not None else None
        cmat = confusion_matrix(self.true_labels, yhat_c)
        
        if plot_roc and self.predicted_probs is not None:
            fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_probs)
            plt.plot(fpr, tpr, label="AUC={:.4}".format(auc_value))
            plt.plot([0, 1], [0, 1], "r--")
            plt.ylabel('True positive rate')
            plt.xlabel('False positive rate')
            plt.legend(loc='lower right')
            plt.show()
        
        return auc_value, acc, cmat
