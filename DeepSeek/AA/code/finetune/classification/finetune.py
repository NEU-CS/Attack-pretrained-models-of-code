import os
import sys
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftType,
    TaskType,
    PeftModelForSequenceClassification
)
from transformers import CodeLlamaTokenizer , LlamaForSequenceClassification ,get_linear_schedule_with_warmup
import evaluate
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.functional import F
from torch.cuda.amp import autocast as autocast,GradScaler
import pandas as pd
from sklearn.metrics import f1_score






device = "cuda" if torch.cuda.is_available() else "cpu"
base_model =  "codellama/CodeLlama-7b-hf"
device_map = "auto"
num_epochs = 20
lr = 5e-5
train_batch_size = 2
eval_batch_size = 1
peft_type = PeftType.LORA
config = LoraConfig(
        r=8,
        lora_alpha=16,
        inference_mode=False,
        lora_dropout=0.1,
        task_type=TaskType.SEQ_CLS,
        target_modules=[
        "q_proj",
        "v_proj",
    ],
    )


tokenizer = CodeLlamaTokenizer.from_pretrained(base_model,model_max_length = 1024)
tokenizer.pad_token = -1

def compute_metrics(eval_pred):
        metirc = evaluate.load("accuracy")
        logits , labels = eval_pred
        predictions = np.argmax(logits,axis=-1)
        return metirc.compute(predictions=predictions,references=labels)


def tokenize_function(examples):
        return tokenizer(examples["text"],truncation = True) 

def collate_fn(examples):
    return tokenizer.pad(examples, padding="max_length", return_tensors="pt")

datasets = load_dataset("csv", data_files={"train":"train.csv","test":"valid.csv"})
tokenized_dataset = datasets.map(tokenize_function,batched=True,remove_columns="text").rename_column("label","labels")
train_dataloader = DataLoader(tokenized_dataset["train"],shuffle=True,collate_fn=collate_fn,batch_size = train_batch_size)
eval_dataloader = DataLoader(tokenized_dataset["test"] , collate_fn=collate_fn,batch_size = eval_batch_size)

model = LlamaForSequenceClassification.from_pretrained(
        base_model,
        load_in_8bit = True,
        torch_dtype = torch.float16,
        num_labels = 66,
        device_map = device_map
    )
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)
model.print_trainable_parameters()
print(model)

optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)
sclaer = GradScaler()

model.to(device)
max_eval_acc = 0
iter_to_accumlate = 4
epochloss = []
trainlogdf = pd.DataFrame(columns=["step","trainloss","validloss","acc","f1-score"])
rowindex = 0
for epoch in range(num_epochs):
    model.train()
    allloss = 0
    for step,batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        with autocast():
            outputs = model(**batch)
        loss = outputs.loss/iter_to_accumlate
        sclaer.scale(loss).backward()
        allloss += loss.item()
        trainlogdf.loc[rowindex] = [rowindex,loss.item(),None,None,None]
        rowindex += 1
        epochloss.append(loss.item())
        if (step+1)%iter_to_accumlate==0:
            sclaer.step(optimizer)
            lr_scheduler.step()
            sclaer.update()
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
        pred = torch.argmax(F.softmax(output.logits,dim=1),dim=1)
        preds += pred
        count += int(batch['labels'].cpu() == pred.cpu())
    eval_acc = count/132
    trainlogdf.loc[rowindex-1,"validloss"] = validloss/132
    trainlogdf.loc[rowindex-1,"acc"] = eval_acc
    trainlogdf.loc[rowindex-1,"f1-score"] = f1_score(np.array(batch['labels'].cpu()),np.array(pred.cpu()),average="macro")
    print("epoch ",epoch,"acc ",eval_acc)
    if eval_acc > max_eval_acc:
        max_eval_acc = eval_acc
        model.save_pretrained("ljcoutputdir")
        torch.save(get_peft_model_state_dict(model),os.path.join("checkpoint","model.bin"))
        torch.save(optimizer.state_dict(),os.path.join("checkpoint","optimizer.bin"))
        torch.save(sclaer.state_dict(),os.path.join("checkpoint","sclaer.bin"))
        torch.save(lr_scheduler.state_dict(),os.path.join("checkpoint","lr_scheduler.bin"))