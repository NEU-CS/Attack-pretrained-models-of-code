# coding = utf-8
# Author:Liu Jincheng
# Date: 2023.4.25
# Email: liujinchengNEU@outlook.com
'''
fine-tune CodeBert
'''

from transformers import RobertaConfig,RobertaModel,RobertaTokenizer,RobertaForSequenceClassification,GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import random
from transformers import TrainingArguments,Trainer
import numpy as np
import evaluate
import torch
import logging

logger = logging.getLogger(__name__)

data_path = os.path.join("..","..","dataset","data_folder","processed_gcjpy")
model_name = "microsoft/codebert-base"
model = RobertaForSequenceClassification.from_pretrained(model_name,num_labels = 66)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def compute_metrics(eval_pred):
    metirc = evaluate.load("accuracy")
    logits , labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    return metirc.compute(predictions=predictions,references=labels)
d

def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length",truncation=True)
        

if __name__ == '__main__':
    torch.cuda.empty_cache()
    dataset = load_dataset("csv",data_files={"train":os.path.join(data_path,"train.csv"),"test":os.path.join(data_path,"valid.csv")})
    tokenized_dataset = dataset.map(tokenize_function,batched=True)
    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42)
    train_args = TrainingArguments(output_dir= "test_trainer",evaluation_strategy="epoch",per_device_train_batch_size=16,per_device_eval_batch_size=8,num_train_epochs=18,learning_rate=5e-5)

    trainer = Trainer(model = model,args = train_args,train_dataset=small_train_dataset,eval_dataset=small_eval_dataset,compute_metrics=compute_metrics)
    trainer.train()
    model.save_pretrained("CodeBERTsaved_models")
    tokenizer.save_pretrained("CodeBERTsaved_models")
    
    