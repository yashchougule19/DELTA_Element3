import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          DistilBertTokenizer, DistilBertForSequenceClassification, 
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset

# Define a Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# A class for handling pre-trained and fine-tuned DistilBERT models
class DistilBERTSentimentAnalyzer:
    def __init__(self, pretrained_model_name, finetuned_model_name, max_length=50):
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
        self.finetuned_tokenizer = DistilBertTokenizer.from_pretrained(finetuned_model_name)
        self.finetuned_model = DistilBertForSequenceClassification.from_pretrained(finetuned_model_name)
        self.max_length = max_length

    def tokenize_data(self, texts, tokenizer):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def set_layer_requires_grad(self, model, layer_name, freeze=True):
        for name, param in model.named_parameters():
            if layer_name in name:
                param.requires_grad = not freeze

    def predict(self, tokens, model):
        with torch.no_grad():
            outputs = model(**tokens)
        return torch.argmax(outputs.logits, dim=1)

    def evaluate_model(self, model, tokens):
        with torch.no_grad():
            outputs = model(**tokens)
        return torch.argmax(outputs.logits, dim=1)

    def fine_tune(self, train_tokens, train_labels, val_tokens, val_labels, epochs=1, batch_size=16):
        # Initial phase: Freeze all layers except the classifier
        self.set_layer_requires_grad(self.finetuned_model, 'classifier', freeze=False)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir='./logs',
        )

        # Define training and evaluation datasets
        train_dataset = SentimentDataset(train_tokens, train_labels)
        val_dataset = SentimentDataset(val_tokens, val_labels)

        # Train DistilBERT model
        trainer = Trainer(
            model=self.finetuned_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        # Gradual unfreezing phase: Unfreeze all layers
        self.set_layer_requires_grad(self.finetuned_model, 'classifier', freeze=False)

        # Optionally: Train with all layers unfrozen
        trainer.train()

# Utility functions for the analysis
def run_pretrained_analysis(analyzer, test_texts):
    tokens = analyzer.tokenize_data(test_texts, analyzer.pretrained_tokenizer)
    return analyzer.predict(tokens, analyzer.pretrained_model)

def run_finetuned_analysis(analyzer, train_texts, train_labels, val_texts, val_labels, test_texts):
    train_tokens = analyzer.tokenize_data(train_texts, analyzer.finetuned_tokenizer)
    val_tokens = analyzer.tokenize_data(val_texts, analyzer.finetuned_tokenizer)
    test_tokens = analyzer.tokenize_data(test_texts, analyzer.finetuned_tokenizer)

    # Convert labels to tensors
    train_labels_tensor = torch.tensor(list(map(int, train_labels)), dtype=torch.long)
    val_labels_tensor = torch.tensor(list(map(int, val_labels)), dtype=torch.long)

    # Fine-tune the model
    analyzer.fine_tune(train_tokens, train_labels_tensor, val_tokens, val_labels_tensor)

    # Evaluate on the test set
    return analyzer.evaluate_model(analyzer.finetuned_model, test_tokens)
