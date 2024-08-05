import torch
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaForTokenClassification
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate a RoBERTa model for token classification.")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
parser.add_argument("--num_labels", type=int, default=9, help="Number of labels.")
args = parser.parse_args()

# Load the dataset
dataset = load_dataset("conll2003")

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_version = 'roberta-base'
num_labels = args.num_labels  # Update with the correct number of labels in your dataset

# Function to add encodings
def add_encodings(example):
    encodings = tokenizer(example['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
    labels = example['ner_tags'] + [0] * (tokenizer.model_max_length - len(example['ner_tags']))
    return {**encodings, 'labels': labels}

# Apply the function to the dataset
dataset = dataset.map(add_encodings)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create label dictionaries
labels = dataset['train'].features['ner_tags'].feature
label2id = {k: labels.str2int(k) for k in labels.names}
id2label = {v: k for k, v in label2id.items()}

# Initialize the model
model = RobertaForTokenClassification.from_pretrained(roberta_version, num_labels=num_labels)
model.config.id2label = id2label
model.config.label2id = label2id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the optimizer
optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)

# Training settings
n_epochs = args.epochs
train_data = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size)

# Training loop
train_loss = []
for epoch in tqdm(range(n_epochs), desc="Epochs"):
    current_loss = 0
    for i, batch in enumerate(train_data):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        current_loss += loss.item()
        if (i + 1) % 100 == 0:  # Print process after every 100 iterations
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(current_loss / 100)
            print(f"Epoch {epoch + 1}, Iteration {i + 1}, Loss: {current_loss / 100:.4f}")
            current_loss = 0
    optimizer.step()
    optimizer.zero_grad()

# Plot training loss
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_loss)
ax.set_ylabel('Loss')
ax.set_xlabel('Iterations (100 examples)')
fig.tight_layout()
plt.show()

# Evaluation
model.eval()
test_data = torch.utils.data.DataLoader(dataset['test'], batch_size=args.batch_size)

# Collect predictions and true labels for metrics calculation
true_labels = []
pred_labels = []
for i, batch in enumerate(tqdm(test_data, desc="Testing Batches")):
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
    s_lengths = batch['attention_mask'].sum(dim=1)
    for idx, length in enumerate(s_lengths):
        true_values = batch['labels'][idx][:length].cpu().numpy()
        pred_values = torch.argmax(outputs.logits, dim=2)[idx][:length].cpu().numpy()
        true_labels.extend(true_values)
        pred_labels.extend(pred_values)

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=labels.names))

# Create confusion matrix
confusion = torch.zeros(num_labels, num_labels)
for true, pred in zip(true_labels, pred_labels):
    confusion[true][pred] += 1

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(confusion.numpy(), cmap='Blues')

labels = list(label2id.keys())
ids = np.arange(len(labels))
ax.set_ylabel('True Labels', fontsize='x-large')
ax.set_xlabel('Pred Labels', fontsize='x-large')
ax.set_xticks(ids)
ax.set_xticklabels(labels)
ax.set_yticks(ids)
ax.set_yticklabels(labels)
fig.tight_layout()
plt.show()