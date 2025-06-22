# train_bom_ner.py
"""
Fine-tune a Hugging Face NER model (e.g., bert-base-cased) for BOM token classification using CoNLL format data.
Usage:
    python train_bom_ner.py --train_file bom_labeling_template.conll --output_dir ./bom_ner_model
"""
import argparse
from datasets import load_dataset, ClassLabel, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import classification_report, f1_score

def get_label_list(train_file):
    labels = set()
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                _, label = line.strip().split()
                labels.add(label)
    return sorted(labels)

def parse_conll_file(filepath):
    """Parse a simple CoNLL file into a list of dicts with 'tokens' and 'ner_tags'."""
    sentences = []
    tokens = []
    ner_tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if tokens:
                    sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
                    tokens = []
                    ner_tags = []
                continue
            splits = line.split()
            if len(splits) == 2:
                token, tag = splits
                tokens.append(token)
                ner_tags.append(tag)
    if tokens:
        sentences.append({'tokens': tokens, 'ner_tags': ner_tags})
    return sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    args = parser.parse_args()

    label_list = get_label_list(args.train_file)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    # Replace dataset loading with custom parsing
    samples = parse_conll_file(args.train_file)
    dataset = Dataset.from_list(samples)
    # Use add_prefix_space=True for RoBERTa
    tokenizer_kwargs = {}
    if 'roberta' in args.model_name.lower():
        tokenizer_kwargs['add_prefix_space'] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)

    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(example['tokens'], truncation=True, is_split_into_words=True)
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(label2id[example['ner_tags'][word_idx]])
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels, batched=False)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        save_steps=100,
        save_total_limit=1,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
