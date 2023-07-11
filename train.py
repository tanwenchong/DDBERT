import numpy as np
import pandas as pd
import torch
import argparse
import torch.utils.data as Data
from transformers import AutoTokenizer, RobertaForSequenceClassification
parser = argparse.ArgumentParser()
parser.add_argument('-i','--train_file')
parser.add_argument('-l','--train_label')
parser.add_argument('-v','--valid_file')
parser.add_argument('-t','--test_file')
args = parser.parse_args()
train_file=args.train_file
train_label=args.train_label
valid_file=args.valid_file
test_file=args.test_file

def get_score(file):
    scores=dict()
    with open(file,'r') as f:
        mark=False
        for line in f:
            if mark==False:
                mark=True
                continue
            else:
                score,name=line.split(',')
                scores[name[:-3]]=abs(float(score))
    return scores

def get_smile(file):
    smiles=dict()
    with open(file,'r') as f:
        mark=False
        for line in f:
            smile,name=line.split(' ')
            smiles[name[:-3]]=smile
    return smiles
  
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
        inputs["labels"] = torch.tensor(self.data.iloc[idx]["labels"], dtype=torch.float).unsqueeze(0)
        return inputs
      
class MyData:
    def __init__(self, i_data):
        self.data = i_data

    def get_split(self, train_ratio=0.5, valid_ratio=0.05, seed=None):
        n = len(self.data)
        indices = np.arange(n)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        train_size = int(train_ratio * n)
        valid_size = int(valid_ratio * n)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size+valid_size]
        test_indices = indices[train_size+valid_size:]
        i_train_data = self.data.iloc[train_indices].reset_index(drop=True)
        i_valid_data = self.data.iloc[valid_indices].reset_index(drop=True)
        i_test_data = self.data.iloc[test_indices].reset_index(drop=True)
        return i_train_data, i_valid_data, i_test_data

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-10M-MLM",num_labels=1)
learning_rate=5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=.0004)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_smiles=get_smile(train_file)
train_smiles=get_score(train_label)
df_m=pd.DataFrame.from_dict(train_smiles, orient='index',columns=['smiles'])
df_s=pd.DataFrame.from_dict(train_smiles, orient='index',columns=['scores'])
df=pd.concat([df_m,df_s],axis=1,join='inner')
df=df.sort_values(by='scores')
df['labels']=df['scores'].gt(df.iloc[int(len(df)*0.99)]['scores'])
df=MyData(df)

data = train_data[['smiles', 'labels']]
valid = valid_data[['smiles', 'labels']]
test = test_data[['smiles', 'labels']]
train_dataset = Input(data, tokenizer, 150)
validation_dataset = Input(valid, tokenizer, 150)

train_loader = Data.DataLoader(dataset=train_dataset,batch_size=50,shuffle=False,num_workers=0,)
valid_loader = Data.DataLoader(dataset=validation_dataset,batch_size=50,shuffle=False,num_workers=0,)

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

for epoch in range(30):
    train_loss =train(model,optimizer,train_loader)
    print(epoch,train_loss)
    valid_loss=train(model,optimizer,valid_loader)
    print(valid_loss)






















