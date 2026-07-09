import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import os

os.environ["WANDB_DISABLED"] = "true"

def train():
    label_map = {
        'true': 0, 'mostly-true': 0, 'half-true': 0,
        'false': 1, 'barely-true': 1, 'pants-fire': 1
    }

    def process_file(path):
        df = pd.read_csv(path, sep='\t', header=None, usecols=[1, 2], names=['label', 'statement'])
        df['statement'] = df['statement'].fillna('')
        df['label'] = df['label'].map(label_map)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        return df

    print("Loading data for Linguistic Finetuning...")
    train_df = process_file('data/liar/train.tsv')
    valid_df = process_file('data/liar/valid.tsv')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['statement'], truncation=True, padding='max_length', max_length=128)

    print("Tokenizing data...")
    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    valid_ds = Dataset.from_pandas(valid_df).map(tokenize, batched=True)

    train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    valid_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, predictions)}

    args = TrainingArguments(
        output_dir='core/models/linguistic_model',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics
    )

    print("Starting Training (This might take a while on CPU)...")
    trainer.train()
    
    # Save Model
    model.save_pretrained('core/models/linguistic_model')
    tokenizer.save_pretrained('core/models/linguistic_model')
    print("Optimization Complete: Linguistic Core Finetuned on LIAR Dataset!")

if __name__ == '__main__':
    train()
