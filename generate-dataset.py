from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
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

print("Building dataset...")

N = 60000
generator = YaraTemplateGenerator()
single_strings     = generator.gen_single_strings(N)
ds = single_strings
d = pd.DataFrame(ds)
d.to_pickle("dataset.pkl")
