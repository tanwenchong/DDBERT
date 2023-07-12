import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import metrics.accuracy_score

test_smiles=test['smiles']
predictions = []
for smiles in test_smiles:
    inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=150).to(device) 
    with torch.no_grad():
        outputs = model(**inputs).logits
    predict = outputs.squeeze().item()
    predictions.append(predict)
