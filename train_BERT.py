import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset

# Define constants
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
LABELS = ["suicida", "no_suicida"]
BATCH_SIZE = 68
EPOCHS = 2
OUTPUT_DIR = './results'

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the dataset class
class EmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Parameters:
    - eval_pred: a tuple (predictions, labels)

    Returns:
    - dict: dictionary with calculated metrics
    """
    logits, labels = eval_pred
    # Convert logits to tensor if they are not already
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
        
    predictions = torch.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(train_file, val_file, test_file, model_name=MODEL_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, output_dir=OUTPUT_DIR):
    """
    Train a BERT model using provided training, validation, and test datasets.

    Parameters:
    - train_file: str, path to the training CSV file.
    - val_file: str, path to the validation CSV file.
    - test_file: str, path to the test CSV file.
    - model_name: str, name of the BERT model to use.
    - batch_size: int, batch size for training and evaluation.
    - epochs: int, number of training epochs.
    - output_dir: str, path to save checkpoints and final model.

    Returns:
    - trainer: Trainer object after training.
    """
    
    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # Ensure correct label mapping
    train_df['label'] = train_df['label'].apply(lambda x: LABELS.index(x))
    val_df['label'] = val_df['label'].apply(lambda x: LABELS.index(x))
    test_df['label'] = test_df['label'].apply(lambda x: LABELS.index(x))

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Create datasets
    train_dataset = EmotionsDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = EmotionsDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
    test_dataset = EmotionsDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

    # Define model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS))
    model.to(device)  # Move model to GPU

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",        # Save checkpoints at the end of each epoch
        load_best_model_at_end=True,  # Load best model at the end of training
        save_total_limit=2,           # Limit the total number of checkpoints
        fp16=True if torch.cuda.is_available() else False,
        metric_for_best_model="accuracy", # Metric to use for selecting the best model
    )

    # Check for existing checkpoints
    last_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith('checkpoint')]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getctime)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Add compute metrics
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Add early stopping
    )

    # Train model
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Evaluate model on the validation set
    val_predictions = trainer.predict(val_dataset)
    val_preds = torch.argmax(torch.tensor(val_predictions.predictions), axis=1)
    print("Validation Classification Report:")
    print(classification_report(val_df['label'], val_preds.cpu(), target_names=LABELS))

    # Evaluate model on the test set
    test_predictions = trainer.predict(test_dataset)
    test_preds = torch.argmax(torch.tensor(test_predictions.predictions), axis=1)
    print("Test Classification Report:")
    print(classification_report(test_df['label'], test_preds.cpu(), target_names=LABELS))  # Move predictions to CPU

    return trainer

# Example usage
trainer = train_model('train_split.csv', 'validation_split.csv', 'test_split.csv')

trainer.save_model("./trained_model")

import pickle

pickle.dump(trainer,"./trained_model.pkl")