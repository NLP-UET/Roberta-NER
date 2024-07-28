# Exploring the Roberta Model and its Application for Named Entity Recognition

## Introduction
This project aims to explore the Roberta model and its application in the task of Named Entity Recognition (NER). Roberta is a variant of BERT with improved performance on various natural language processing (NLP) tasks.

## Team Members
- **Nguyen Viet Bac** - 22022511
- **Dang Van Khai** - 22022550
- **Duong Minh Duc** - 22022606

## Project Objectives
1. Understand the Roberta model.
2. Apply the Roberta model to the task of Named Entity Recognition.
3. Evaluate the model's performance on standard datasets.

## Content
### 1. Introduction to the Roberta Model
- Overview of the Roberta model.
- Key features and improvements over the BERT model.

### 2. Data Preparation
- Collect and preprocess data for the NER task.
- Split the dataset into training, validation, and test sets.

### 3. Model Training
- Install necessary libraries.
- Train the Roberta model on the NER dataset.
- Save and load the trained model.

### 4. Evaluation and Testing
- Evaluate the model on the test set.
- Analyze results and provide insights.

### 5. Conclusion
- Summarize the achieved results.
- Suggest future directions for the project.

## System Requirements
- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Other libraries: pandas, numpy, scikit-learn, etc.

## Setting up

```bash
import os
os.environ['PARAM_SET'] = 'base' # change to 'large' to use the large architecture

# we use pip 23.1 to run 
pip install pip==23.1
# clone the repo
git clone https://github.com/NLP-UET/Roberta-NER.git
pip install -r ./Roberta-NER/requirements.txt
gdown --id 1uC7UYA-BDg-dJYB_C6aphyJ5IWweVzXE  # base model
gdown --id 15HA6Iq5Gld2XXV27lmOOa3KlK-DV_gTq  # large model
tar -xzvf xlmr.base.tar.gz -C ./Roberta-NER/pretrained_models/
tar -xzvf xlmr.large.tar.gz -C ./Roberta-NER/pretrained_models/
rm -r xlmr.base.tar.gz
rm -r xlmr.large.tar.gz
```

## Training arguments:
```bash
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or
                        other data files) for the task.
  --pretrained_path PRETRAINED_PATH
                        pretrained XLM-Roberta model path
  --task_name TASK_NAME
                        The name of the task to train.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        WordPiece tokenization. Sequences longer than this
                        will be truncated, and sequences shorter than this
                        will be padded.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval or not.
  --eval_on EVAL_ON     Whether to run eval on the dev set or test set.
  --do_lower_case       Set this flag if you are using an uncased model.
  --train_batch_size TRAIN_BATCH_SIZE
                        Total batch size for training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Total batch size for eval.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --warmup_proportion WARMUP_PROPORTION
                        Proportion of training to perform linear learning rate
                        warmup for. E.g., 0.1 = 10% of training.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --no_cuda             Whether not to use CUDA when available
  --seed SEED           random seed for initialization
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --fp16                Whether to use 16-bit float precision instead of
                        32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --loss_scale LOSS_SCALE
                        Loss scaling to improve fp16 numeric stability. Only
                        used when fp16 set to True. 0 (default value): dynamic
                        loss scaling. Positive power of 2: static loss scaling
                        value.
  --dropout DROPOUT     training dropout probability
  --freeze_model        whether to freeze the XLM-R base model and train only
                        the classification heads
```
## How to run

For example:
```bash
python ./Roberta-NER/main.py \
    --data_dir ./Roberta-NER/data/coNLL-2003/ \
    --task_name ner \
    --output_dir ./Roberta-NER/model_dir/ \
    --max_seq_length 16 \
    --num_train_epochs 1 \
    --do_eval \
    --warmup_proportion 0.1 \
    --pretrained_path ./Roberta-NER/pretrained_models/xlmr.$PARAM_SET \
    --learning_rate 0.00007 \
    --do_train \
    --eval_on test \
    --train_batch_size 4 \
    --dropout 0.2
```
If you want to use the XLM-R model's outputs as features without finetuning, Use the `--freeze_model` argument.

By default, the best model on the validation set is saved to `args.output_dir`. This model is then loaded and tested on the test set, if `--do_eval` and `--eval_on test`.

### Predicting
To predict named entities in any text using the trained model, you can run the following command:
```bash
python ./Roberta-NER/predict.py --pretrained_path ./Roberta-NER/pretrained_models/xlmr.base --output_dir ./Roberta-NER/model_dir/ --text "your input text"
```
Simply replace `"your input text"` with the text you want to analyze.

## Contribution
We welcome contributions from the community. Please create an issue or submit a pull request so we can review and integrate it into the project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.