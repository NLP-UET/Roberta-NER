import argparse
import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from sklearn.metrics import classification_report
import warnings
import os

# Suppress warnings related to tags not being recognized
warnings.filterwarnings("ignore", message=".*seems not to be NE tag.*")

# Define the labels for the CoNLL-2003 dataset
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
main_labels = ["PER", "ORG", "LOC", "MISC"]  # Main labels excluding 'O'

# Define a function to convert sub-labels to main labels
def convert_to_main_labels(labels):
    label_map = {
        "B-PER": "PER", "I-PER": "PER",
        "B-ORG": "ORG", "I-ORG": "ORG",
        "B-LOC": "LOC", "I-LOC": "LOC",
        "B-MISC": "MISC", "I-MISC": "MISC",
        "O": "O"
    }
    return [label_map.get(label, "O") for label in labels]

# Define a function to compute metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]

    # Convert to main labels for the classification report
    true_labels_main = [convert_to_main_labels(label) for label in true_labels]
    true_predictions_main = [convert_to_main_labels(prediction) for prediction in true_predictions]

    results = metric.compute(predictions=true_predictions_main, references=true_labels_main)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def log(self, logs):
        super().log(logs)
        if 'epoch' in logs:
            print(f"Epoch: {logs['epoch']}, Loss: {logs['loss']:.4f}")

def main(args):
    # Load the CoNLL-2003 dataset
    datasets = load_dataset("conll2003")

    # Load the tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=len(label_list))

    # Tokenize the data
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], padding='max_length', truncation=True, max_length=128, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            # Pad labels to max length
            label_ids += [-100] * (128 - len(label_ids))
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply tokenization and label alignment to the train and validation sets
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "pos_tags", "chunk_tags", "ner_tags"])

    # Define TrainingArguments with parameters from command line arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "results"),
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )

    # Define the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    true_labels = [label for sublist in [[label_list[l] for l in label if l != -100] for label in labels] for label in sublist]
    true_predictions = [label for sublist in [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)] for label in sublist]

    # Convert to main labels for the classification report
    true_labels_main = convert_to_main_labels(true_labels)
    true_predictions_main = convert_to_main_labels(true_predictions)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(args.output_dir, "result")
    os.makedirs(results_dir, exist_ok=True)

    # Save the classification report
    report = classification_report(true_labels_main, true_predictions_main, labels=main_labels, zero_division=0)
    report_filename = os.path.join(results_dir, "roberta_base_classification_report.txt")
    with open(report_filename, "w") as f:
        f.write(report)
    
    print(f"Classification report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./Roberta-NER", help="Directory to save the results and model checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training")

    args = parser.parse_args()
    main(args)