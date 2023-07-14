import pandas as pd
import numpy as np
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
	
def get_weight(df,ratio=0.99):
    if int(df)==False:
        return int(1/ratio)
    else:
        return int(1/(1-ratio))
	    
def get_dataframe(smiles,scores,ratio):
    df_m=pd.DataFrame.from_dict(smiles, orient='index',columns=['smiles'])
    df_s=pd.DataFrame.from_dict(scores, orient='index',columns=['scores'])
    df=pd.concat([df_m,df_s],axis=1,join='inner')
    df=df.sort_values(by='scores')
    df['labels']=df['scores'].gt(df.iloc[int(len(df)*ratio)]['scores'])	
    df['weight']=df['labels'].map(get_weight)
    return df
	    
def get_sampler(df,ratio,num_samples):
	weight=np.array(df[['weight']]).tolist()
	weight=np.squeeze(weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, num_samples, replacement=True)
	return sampler







