import matplotlib.pyplot as plt
from sklearn import metrics 
import numpy as np
import pandas as pd
import torch
import argparse
import torch.utils.data as Data
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-t','--test_file')
parser.add_argument('-l','--test_label')
parser.add_argument('-m','--model')

args = parser.parse_args()
test_file=args.test_file
test_label=args.test_label
model_dict=args.model

class Input(Dataset):
    def __init__(self, i_data, i_tokenizer, i_max_length):
        self.data = i_data
        self.tokenizer = i_tokenizer
        self.max_length = i_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile = self.data.iloc[idx]['smiles']
        inputs = self.tokenizer(smile, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        inputs["labels"] = torch.tensor(self.data.iloc[idx]["labels"], dtype=torch.long).unsqueeze(0)
        return inputs

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-10M-MLM",num_labels=2)
model.load_state_dict(torch.load(model_dict))
ratio=0.99
test_smiles=get_smile(test_file)
test_scores=get_score(test_label)
test_data=get_dataframe(test_smiles,test_scores,ratio)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
test=test_data[['smiles','labels']]
#print(test_data[['weight']])
predictions = []
labels=[]
i=0
test_dataset = Input(test, tokenizer, 150)

test_loader = Data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=0,sampler = get_sampler(test_data,ratio,len(test_dataset)))
with torch.no_grad():
    for batch in test_loader:
        i+=1
        if i==1000:
          break
        outputs=model(batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
        predict = outputs.logits.argmax(axis=1)
        predictions.append(int(predict.cpu()))
        labels.append(int(batch['labels']))
        #print(labels)
report = metrics.classification_report(test_data['labels'],predictions)
fpr, tpr, thresholds = metrics.roc_curve(test_data['labels'],predictions)
auc = metrics.auc(fpr,tpr)
plt.plot(fpr,tpr,'*-')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
print(report)














