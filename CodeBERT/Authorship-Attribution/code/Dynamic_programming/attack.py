# coding=utf-8
# @Time    : 2024.1.21
# @Author  : Jincheng Liu
# @Email   : jinchengliu@smail.nju.edu.cn
# @File    : attacker.py
'''For attacking CodeLLaMA models by beam search to get a relativly good result'''
import sys
import os

sys.path.append('../../../../')
sys.path.append('../../../../python_parser')
retval = os.getcwd()

import json
import logging
import argparse
import warnings
import torch
import time
from utils import Recorder,set_seed
from attacker import Attacker
from transformers import RobertaTokenizer, RobertaForSequenceClassification,pipeline,RobertaForMaskedLM
from datasets import load_dataset
from peft import PeftModelForSequenceClassification
from torch.functional import F
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\
logger = logging.getLogger(__name__)






class myclassifier():
    def __init__(self,classifier):
        self.classifier = classifier
        self.querytimes = 0

    def predict(self,code):
        if type(code) == list:
            self.querytimes += len(code)
        elif type(code) == str:
            self.querytimes += 1
        return self.classifier(code)
    
    def query(self):
        return self.querytimes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default="../finetune/CodeBERTsaved_models", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="../finetune/CodeBERTsaved_models", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--beam_size",default=1,type=int)
    parser.add_argument("--csv_store_path", type=str,
                        help="Path to store the CSV file")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--use_replace",action='store_true')
    parser.add_argument("--use_insert",action='store_true')
    parser.add_argument("--sort_substitute",action="store_true")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    
    
    
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels = 66)  
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')      
    model.to(args.device)

    def bertpipeline():
        def cls(example):
            if type(example) != list:
                example = [example]
            batchsize = 32
            score = []
            label = []
            for i in range(len(example)//batchsize+1 if len(example)%batchsize != 0 else len(example)//batchsize):
                input = tokenizer(example[i*batchsize:batchsize*(i+1)],padding="max_length",truncation=True,return_tensors = "pt")
                input.to(args.device)
                with torch.no_grad():
                    output = model(**input)
                s = F.softmax(output.logits.float(),dim=1)
                s,l = torch.max(s,dim=1)
                score += s
                label += l
            ret = []
            for i in range(len(label)):
                ret.append({"label":"LABEL_"+str(int(label[i])),"score":float(score[i])})
            return ret
        
        return cls
    classifier = bertpipeline()
    classifier = myclassifier(classifier)

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    #codebert_mlm.to(args.device) 

    ## Load Dataset
    eval_dataset = load_dataset("csv", data_files = args.eval_data_file)
    eval_dataset = eval_dataset['train']
    file_type = args.eval_data_file.split('/')[-1].split('.')[0] # valid
    folder = '/'.join(args.eval_data_file.split('/')[:-1]) # 得到文件目录
    codes_file_path = os.path.join(folder, '{}_subs.jsonl'.format(
                                file_type))
    substs = []
    with open(codes_file_path) as rf:
        for line in rf:
            item = json.loads(line.strip())
            substs.append(item["substitutes"])
    assert(len(eval_dataset) == len(substs))

    success_attack = 0
    total_cnt = 0
    recoder = Recorder(args.csv_store_path)
    attacker = Attacker(args, classifier,model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    query_times = 0
    greedy_query_times = 0
    ga_query_times = 0
    all_sub_index = 0
    all_acc_nums = 1
    for index, example in enumerate(eval_dataset):
        example_start_time = time.time()
        code = example['text']
        true_label = example['label']
        subs = substs[index]
        print("enter")
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words,Suctype,insertwords,most_index,acc_nums = attacker.greedy_attack(code,true_label, subs,args.use_replace,args.use_insert,args.beam_size,args.sort_substitute)
        
        all_acc_nums += acc_nums
        greedy_query_times = classifier.query() - query_times
        all_sub_index += most_index
        ganb_changed_var = 0
        ganb_changed_pos = 0
        attack_type = "Greedy"
        if is_success == -1 and args.use_ga:
            # 如果不成功，则使用gi_attack
            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, ganb_changed_var, ganb_changed_pos, replaced_words = attacker.ga_attack(code,true_label, subs, initial_replace=replaced_words)
            attack_type = "GA"
            ga_query_times = classifier.query() - greedy_query_times - query_times

        example_end_time = (time.time()-example_start_time)/60
        
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time()-start_time)/60, 2), "min")
        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        print("Query times in this attack: ", classifier.query() - query_times)
        print("Greedy query times:",greedy_query_times)
        print("Greedy sub index:",most_index)
        print("Ga query times:",ga_query_times)
        print("All Query times: ", classifier.query())
        print("All sub index:",all_sub_index)
        print("Avg sub index",all_sub_index//all_acc_nums )
        query_times = classifier.query()
        
        if is_success >= -1 :
            # 如果原来正确
            total_cnt += 1
        if is_success == 1:
            success_attack += 1
            
        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print("Successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)
        print()
        recoder.write(index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info, ganb_changed_var,ganb_changed_pos,attack_type, classifier.query() - query_times, example_end_time,Suctype,insertwords,)

    
    
if __name__ == '__main__':
    main()
