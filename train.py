from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, EarlyStoppingCallback
from transformers import TrainingArguments
from transformers import default_data_collator
from functools import partial

from BERTClassifier import JapaneseBERTClassfier


def makeDataset(datset_json_path, tokenizer, max_size, test_size=0.2):
    data = pd.read_json(datset_json_path)
    train, valid = train_test_split(data, test_size=test_size)

    print('train', train.shape)
    print('valid', valid.shape)

    ds_train = Dataset.from_pandas(train)
    ds_valid = Dataset.from_pandas(valid)

    dataset = DatasetDict({
        "train": ds_train,
        "validation": ds_valid,
    })

    text2token_partial = partial(text2token, tokenizer=tokenizer, max_size=max_size)
    dataset = dataset.map(text2token_partial, batched=True)

    print(dataset)
    return dataset


def text2token(data, tokenizer, max_size=450):
    texts = [q.strip() for q in data["text"]]
    inputs = tokenizer(
        texts,
        max_length=max_size,
        truncation=True,
        padding=True,
    )

    inputs['labels'] = torch.tensor(data['label'])

    return inputs


def train(bert_model, training_args, tokenized_data, data_collator, tokenizer):
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],
    )

    print('start training')
    trainer.train()
    print('end training')


def main(args):
    json_path = args.json_path
    bert_model_path = args.bert_model_path
    tokenizer_path = args.tokenizer_path
    device = args.device
    num_labels = args.num_labels
    output_dir = args.output_dir

    bert_classifiler_model = JapaneseBERTClassfier(bert_model_path, tokenizer_path, device, num_labels)

    model = bert_classifiler_model.model
    tokenizer = bert_classifiler_model.tokenizer

    dataset = makeDataset(json_path, tokenizer, max_size=450)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=200,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    data_collator = default_data_collator

    train(model, training_args, dataset, data_collator, tokenizer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="data.json")
    parser.add_argument("--bert_model_path", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--tokenizer_path", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_labels", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    main(args)
