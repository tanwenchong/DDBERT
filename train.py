import numpy as np
import pandas as pd
import torch
import argparse
import torch.utils.data as Data
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('-t','--train_file')
parser.add_argument('-a','--train_label')
parser.add_argument('-v','--valid_file')
parser.add_argument('-b','--valid_label')
args = parser.parse_args()
train_file=args.train_file
train_label=args.train_label
valid_file=args.valid_file
valid_label=args.valid_label
model_name='DDBERT'  
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

ratio=0.99
batch_size=100
learning_rate=5e-5


tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-10M-MLM",num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=.0004)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_smiles=get_smile(train_file)
train_scores=get_score(train_label)
train_data=get_dataframe(train_smiles,train_scores,ratio)
valid_smiles=get_smile(valid_file)
valid_scores=get_score(valid_label)
valid_data=get_dataframe(valid_smiles,valid_scores,ratio)

train = train_data[['smiles', 'labels']]
valid = valid_data[['smiles', 'labels']]


train_dataset = Input(train, tokenizer, 150)
valid_dataset = Input(valid, tokenizer, 150)

train_loader = Data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,num_workers=0,sampler = get_sampler(train_data,ratio,len(train_dataset)))
valid_loader = Data.DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0,sampler =get_sampler(valid_data,ratio,len(valid_dataset)))

def train(model,optimizer,loader):
    total_loss=0
    model.train()
    for batch in loader:
		outputs=model(batch['input_ids'].to(device),labels=batch['labels'].to(device),attention_mask=batch['attention_mask'].to(device))
		loss=outputs.loss
		total_loss+=loss.item()
		loss.backward()
		optimizer.step()
		#scheduler.step()
		optimizer.zero_grad()
    return total_loss

valid_loss=1000

for epoch in range(10):
	train_loss =train(model,optimizer,train_loader)/ len(train_loader)
	print('epoch={},train_loss={}'.format(epoch,train_loss))
	with torch.no_grad():
		vloss=train(model,optimizer,valid_loader)/len(valid_loader)
	if vloss<valid_loss:
		valid_loss=vloss
		torch.save(model.state_dict(),model_name)
		print('epoch={},valid_loss={}'.format(epoch,vloss))
	else:
		print('epoch={},valid_loss={}'.format(epoch,vloss))



























