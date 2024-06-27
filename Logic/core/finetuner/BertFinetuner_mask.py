import os
import sys
current_script_path = os.path.abspath(__file__)
finetuner = os.path.dirname(current_script_path)
core_dir = os.path.dirname(finetuner)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


# Tools
from tqdm import tqdm

# Base
import pandas as pd
import numpy as np
import json

# Torch
import torch
import wandb
wandb.init(mode='disabled')
 
# sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Hugging Face
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, EvalPrediction

# Visualization
import matplotlib.pyplot as plt


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5, checkpoint='bert-base-uncased'):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        # TODO: Implement initialization logic
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.checkpoint = checkpoint

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open(self.file_path, 'r') as f:
            self.dataset = json.load(f)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # TODO: Implement genre filtering and visualization logic
        genre_counts = {}
        for movie in tqdm(self.dataset, desc='Counting genres'):
            if movie['genres']:
                for genre in movie['genres']:
                    if genre not in genre_counts:
                        genre_counts[genre] = 0
                    genre_counts[movie['genres'][0]] += 1

        top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:self.top_n_genres]

        plt.bar(range(len(top_genres)), [genre_counts[genre] for genre in top_genres], align='center')
        plt.xticks(range(len(top_genres)), top_genres, rotation=45)
        plt.show()

        self.genre2idx = {genre: idx for idx, genre in enumerate(top_genres)}
        self.idx2genre = {idx: genre for genre, idx in self.genre2idx.items()}

        self.preprocess_dataset = {}
        for movie in tqdm(self.dataset, desc='Preprocessing dataset'):
            if movie['genres'] and movie['first_page_summary']:
                for genre in movie['genres']:
                    if genre in self.genre2idx:
                        if movie['first_page_summary'] not in self.preprocess_dataset:
                            self.preprocess_dataset[movie['first_page_summary']] = []

                        self.preprocess_dataset[movie['first_page_summary']].append(self.genre2idx[genre])

        for key, value in self.preprocess_dataset.items():
            one_hot = np.zeros(self.top_n_genres)
            for genre in value:
                one_hot[genre] = 1
            self.preprocess_dataset[key] = one_hot
        
    
    def split_dataset(self, test_size=0.1, val_size=0.1):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # TODO: Implement dataset splitting logic
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(list(self.preprocess_dataset.keys()), list(self.preprocess_dataset.values()), test_size=test_size, random_state=42)
        val_size = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=42)

        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)

        train_encoded = self.tokenizer(self.X_train, truncation=True, padding=True, max_length=512, return_tensors='pt')
        val_encoded = self.tokenizer(self.X_val, truncation=True, padding=True, max_length=512, return_tensors='pt')
        test_encoded = self.tokenizer(self.X_test, truncation=True, padding=True, max_length=512, return_tensors='pt')

        self.train_dataset = self.create_dataset(train_encoded, self.y_train)
        self.val_dataset = self.create_dataset(val_encoded, self.y_val)
        self.test_dataset = self.create_dataset(test_encoded, self.y_test)

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # TODO: Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=15, batch_size=16, warmup_steps=500, weight_decay=0.001, learning_rate=4e-5):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # TODO: Implement BERT fine-tuning logic


        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = BertForSequenceClassification.from_pretrained(self.checkpoint, 
                                                              num_labels=self.top_n_genres, 
                                                              id2label=self.idx2genre, 
                                                              label2id=self.genre2idx,
                                                              problem_type="multi_label_classification")

        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(6):
            for param in self.model.bert.encoder.layer[i].parameters():
                param.requires_grad = False


        training_args = TrainingArguments(
            output_dir=project_root+"/models",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            logging_dir=project_root+"/logs",
            # push_to_hub=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics,
            # callbacks=[wandb.integration.HuggingFaceCallback()]
        )

        self.trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # TODO: Implement metric computation logic
        logits, labels = pred.predictions, pred.label_ids
        predictions = (logits >= 0.5).astype(int)

        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        accuracy = accuracy_score(labels, predictions)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # TODO: Implement model evaluation logic

        results = self.trainer.evaluate(self.train_dataset)
        print(f"Train Set Evaluation:\n"
            f"Precision: {results['eval_precision']}\n"
            f"Recall: {results['eval_recall']}\n"
            f"F1: {results['eval_f1']}\n"
            f"Accuracy: {results['eval_accuracy']}\n"
            "-----------------------------------------")
        
        results = self.trainer.evaluate(self.val_dataset)
        print(f"Validation Set Evaluation:\n"
            f"Precision: {results['eval_precision']}\n"
            f"Recall: {results['eval_recall']}\n"
            f"F1: {results['eval_f1']}\n"
            f"Accuracy: {results['eval_accuracy']}\n"
            "-----------------------------------------")
        

        results = self.trainer.evaluate(self.test_dataset)
        print(f"Test Set Evaluation:\n"
            f"Precision: {results['eval_precision']}\n"
            f"Recall: {results['eval_recall']}\n"
            f"F1: {results['eval_f1']}\n"
            f"Accuracy: {results['eval_accuracy']}\n"
            "-----------------------------------------")
        
        

    def classification_report(self, split='test'):

        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")


        # Detailed classification report
        logits, labels = self.trainer.predict(dataset).predictions, dataset.labels
        predictions = (logits >= 0.5).astype(int)
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        print(f"Classification Report for {split.capitalize()} Split:\n", classification_report(labels, predictions, target_names=self.idx2genre.values()))


    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic

        path = project_root+'/models/'+model_name
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        self.model.push_to_hub(model_name)
        self.tokenizer.push_to_hub(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # TODO: Implement item retrieval logic
        item = {}
        item['input_ids'] = self.encodings['input_ids'][idx]
        item['attention_mask'] = self.encodings['attention_mask'][idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)
        