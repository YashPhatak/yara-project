from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tokenizer import tokenizer

import numpy as np
import pandas as pd
import evaluate
import sys
import random
import string
import re
import humanize
import shutil
from yara_rule_gen.generator import *

print("Loading dataset...")

d = pd.read_pickle("dataset.pkl")
dt = Dataset.from_pandas(d)
dataset = DatasetDict({"train": dt})
dataset = dataset["train"].train_test_split(test_size=0.2)

print("Size of training dataset is %s" % humanize.naturalsize(sys.getsizeof(d),binary=True))

print("Loading Tokenizer...")

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



tokenizer.add_tokens(["{","}","\\"])
source_lang = "en"
target_lang = "yara"
prefix = "translate English to YARA: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

print("Tokenizing...")
tokenized_yaras = dataset.map(preprocess_function, batched=True, num_proc=30)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model.resize_token_embeddings(len(tokenizer))

#import IPython
#IPython.embed()
#sys.exit(0)

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_opus_yaras_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)

metric = evaluate.load("sacrebleu")

# normalize var name to adjust the bleu score
def normalize_vars(rule):
    var_names = re.findall("\$([a-zA-Z0-9_ ]*)=",rule)
    for i, var in enumerate(var_names):
        src_var = "$%s" % var
        target_var = "$var%d " % i
        rule = rule.replace(src_var,target_var)
    return rule

def postprocess_text(preds, labels):
    # need to normalize vars here if want more precise bleu score
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_yaras["train"],
    eval_dataset=tokenized_yaras["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training...")

trainer.train()

trainer.save_model("yara-t5")
