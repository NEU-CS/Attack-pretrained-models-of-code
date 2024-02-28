from transformers import RobertaForSequenceClassification,RobertaTokenizer,get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import random
import numpy as np
import evaluate
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.functional import F
from torch.cuda.amp import autocast as autocast,GradScaler
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim import AdamW

data_path = os.path.join("..","dataset","data_folder","processed_gcjpy")
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("CodeBERTsaved_models")

model = RobertaForSequenceClassification.from_pretrained(model_name,num_labels = 66)
train_batch_size = 8
eval_batch_size = 32
lr = 5e-5
num_epochs = 20

def compute_metrics(eval_pred):
        metirc = evaluate.load("accuracy")
        logits , labels = eval_pred
        predictions = np.argmax(logits,axis=-1)
        return metirc.compute(predictions=predictions,references=labels)


def tokenize_function(examples):
        return tokenizer(examples["text"],truncation = True) 

def collate_fn(examples):
    return tokenizer.pad(examples, padding="max_length", return_tensors="pt")


datasets = load_dataset("csv", data_files={"train":os.path.join(data_path,"train.csv"),"test":os.path.join(data_path,"valid.csv")})
tokenized_dataset = datasets.map(tokenize_function,batched=True,remove_columns="text").rename_column("label","labels")
train_dataloader = DataLoader(tokenized_dataset["train"],shuffle=True,collate_fn=collate_fn,batch_size = train_batch_size)
eval_dataloader = DataLoader(tokenized_dataset["test"] , collate_fn=collate_fn,batch_size = eval_batch_size)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)


model.to(device)
max_eval_acc = 0
iter_to_accumlate = 1
epochloss = []
trainlogdf = pd.DataFrame(columns=["step","trainloss","validloss","acc","f1-score"])
rowindex = 0
for epoch in range(num_epochs):
    model.train()
    allloss = 0
    for step,batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss/iter_to_accumlate
        loss.backward()
        allloss += loss.item()
        trainlogdf.loc[rowindex] = [rowindex,loss.item(),None,None,None]
        rowindex += 1
        epochloss.append(loss.item())
        if (step+1)%iter_to_accumlate==0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if (step+1)%(4*iter_to_accumlate) == 0:
            print("epoch",epoch,"step",step,"loss",loss,sep=" ")
            
    print("epoch",epoch,"trainLoss:",allloss/(len(train_dataloader)*train_batch_size))

    count = 0
    model.eval()
    validloss = 0
    preds = []
    labels = []
    for step,batch in enumerate(tqdm(eval_dataloader)):
        labels += batch['labels'].cpu()
        batch.to(device)
        with torch.no_grad():
            output = model(**batch)
        validloss += output.loss.item()
        pred = torch.argmax(F.softmax(output.logits.cpu(),dim=1),dim=1)
        preds += pred
        count += int(sum(batch['labels'].cpu() == pred))
    eval_acc = count/132
    trainlogdf.loc[rowindex-1,"validloss"] = validloss/132
    trainlogdf.loc[rowindex-1,"acc"] = eval_acc
    trainlogdf.loc[rowindex-1,"f1-score"] = f1_score(np.array(labels),np.array(preds),average="macro")
    print("epoch ",epoch,"acc ",eval_acc)
    if eval_acc > max_eval_acc:
        max_eval_acc = eval_acc
        model.save_pretrained("CodeBERTsaved_models2")
        torch.save(model.state_dict(),os.path.join("checkpoint","model.bin"))
        torch.save(optimizer.state_dict(),os.path.join("checkpoint","optimizer.bin"))
        torch.save(lr_scheduler.state_dict(),os.path.join("checkpoint","lr_scheduler.bin"))


trainlogdf.to_csv("trainlog.csv")
tokenizer.save_pretrained("CodeBERTsaved_models")
