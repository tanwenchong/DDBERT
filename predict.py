import os
import glob
import numpy as np
import pandas as pd
import torch
import argparse
import torch.utils.data as Data
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('-f','--foder')
parser.add_argument('-m','--model')
args = parser.parse_args()
foder=args.foder
directory = os.getcwd()
files = glob.glob(directory + foder+"/*")
try:
    os.makedirs(directory+'/preditions')
except:
    print('already exist')

model_dict=args.model
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-10M-MLM",num_labels=2)
model.load_state_dict(torch.load(model_dict))
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

for file in files:
    input_smiles=get_smile(file)
    df_m=pd.DataFrame.from_dict(input_smiles, orient='index',columns=['smiles'])
    df_m=df_m[['smiles']]
    dataset = Input(df_m, tokenizer, 150)
    batch_size=128
    loader = Data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs=model(batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
            predict = outputs.logits.argmax(axis=1)
            predictions.append(int(predict.cpu()))
    predictions=np.array(predictions)
    df_p=pd.Series(predictions,index=df_m.index)
    df=pd.concat([df_m,df_p],axis=1)
    df.to_pickle(directory+'/preditions/'+file)































