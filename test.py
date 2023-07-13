import matplotlib.pyplot as plt
import sklearn.metrics 
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
parser.add_argument('-m','--model')

args = parser.parse_args()
test_file=args.test_file
test_label=args.test_label
model_dict=args.model

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-10M-MLM",num_labels=2)
model.load_state_dict(torch.load(model_dict))

test_smiles=get_smile(test_file)
test_scores=get_score(test_label)
test_data=get_dataframe(test_smiles,test_scores,ratio)

test_smiles=test_data['smiles']

predictions = []
for smiles in test_smiles:
    inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=150).to(device) 
    with torch.no_grad():
        outputs = model(**inputs)
    predict = outputs.logits.argmax(axis=1)
    predictions.append(predict)
    
report = metrics.classification_report(test_data['labels'],predictions)
fpr, tpr, thresholds = metrics.roc_curve(test_data['labels'],predictions)
auc = metrics.auc(fpr,tpr)
plt.plot(fpr,tpr,'*-')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
print(report)














