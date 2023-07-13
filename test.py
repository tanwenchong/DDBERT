import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import metrics.accuracy_score
import numpy as np
import pandas as pd
import torch
import argparse
import torch.utils.data as Data
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('-t','--test_file')
parser.add_argument('-l','--test_label')
args = parser.parse_args()
test_file=args.test_file
test_label=args.test_label


test_smiles=test['smiles']
predictions = []
for smiles in test_smiles:
    inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=150).to(device) 
    with torch.no_grad():
        outputs = model(**inputs).logits
    predict = outputs.squeeze().item()
    predictions.append(predict)
