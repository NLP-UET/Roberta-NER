import torch
from transformers import XLMRobertaTokenizer
from model.xlmr_for_token_classification import XLMRForTokenClassification
from utils.data_utils import NerProcessor, convert_examples_to_features
import argparse
import os
import re

def predict_ner(model, tokenizer, text, label_map, device):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Predict NER labels
    with torch.no_grad():
        logits = model(inputs_ids=input_ids, labels=None, labels_mask=None, valid_mask=None)

    # Convert logits to label IDs
    label_ids = torch.argmax(logits, dim=2).detach().cpu().numpy()

    # Convert label IDs to label names
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [label_map[label_id] for label_id in label_ids[0]]

    # Filter out unwanted tokens and remove prefix "▁"
    result = []
    current_word = ''
    current_label = 'O'
    for token, label in zip(tokens, labels):
        if token in ["<s>", "</s>"]:
            continue
        if token.startswith("▁"):
            if current_word:
                result.append((current_word, current_label))
            current_word = token[1:]
            current_label = label
        else:
            current_word += token
        # If the label is not 'O', update the current label to the new label
        if label != 'O':
            current_label = label
    if current_word:
        result.append((current_word, current_label))

    # Remove punctuation from results
    result = [(word, label) for word, label in result if word]

    # Preserve original formatting for special tokens
    preserved_result = []
    for match in re.finditer(r'\w+[-\w]*', text):
        word = match.group()
        for res_word, label in result:
            if word.replace('-', '').replace('_', '') == res_word:
                preserved_result.append((word, label))
                break
        else:
            preserved_result.append((word, 'O'))
    
    return preserved_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default="./Roberta-NER/pretrained_models/xlmr.base", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--output_dir", default="./Roberta-NER/model_dir/", type=str, required=True, help="Directory to save the model and tokenizer")
    parser.add_argument("--text", type=str, required=True, help="Input text for NER prediction")
    args = parser.parse_args()

    # Load the processor and label map
    processor = NerProcessor()
    label_list = processor.get_labels()
    label_map = {i: label for i, label in enumerate(label_list, 1)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_path)

    # Initialize the model
    model = XLMRForTokenClassification(
        pretrained_path=args.pretrained_path,
        n_labels=len(label_list) + 1,
        hidden_size=768,  # or 1024 depending on your model size
        dropout_p=0.1,
        device=device
    )

    # Load the model state
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.pt')))
    model.to(device)
    model.eval()

    # Predict NER for the input text
    predictions = predict_ner(model, tokenizer, args.text, label_map, device)

    # Print the predictions
    for token, label in predictions:
        print(f"{token}: {label}")

if __name__ == "__main__":
    main()
