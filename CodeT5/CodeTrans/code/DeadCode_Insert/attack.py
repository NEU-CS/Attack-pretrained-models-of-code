# coding=utf-8
# @Time    : 2023.4.25
# @Author  : Liu Jincheng
# @Email   : liujinchengNEU@outlook.com
# @File    : attack.py
'''For attacking CodeT5 models'''
import sys
import os

sys.path.append('../../../../')
sys.path.append('../../../../python_parser')
sys.path.append('../finetune/CodeBLEU')
sys.path.append("../finetune/")
retval = os.getcwd()

import json
import logging
import argparse
import warnings
import torch
import time
from utils import Recorder,set_seed
from attacker import Attacker
from transformers import RobertaTokenizer, T5ForConditionalGeneration,pipeline,RobertaForMaskedLM
from datasets import load_dataset
from torch.functional import F
from tqdm import tqdm
from CodeBLEU.calc_code_bleu import compute_metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\
logger = logging.getLogger(__name__)



class myclassifier():
    def __init__(self,classifier):
        self.classifier = classifier
        self.querytimes = 0

    def predict(self,code):
        if type(code['code']) == list:
            self.querytimes += len(code['code'])
        elif type(code['code']) == str:
            self.querytimes += 1 
        return self.classifier(code)
    
    def query(self):
        return self.querytimes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--sub_data_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default="../finetune/CodeBLEU/CodeT5ForCodeTrans", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="../finetune/CodeBLEU/CodeT5ForCodeTrans", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--csv_store_path", type=str,
                        help="Path to store the CSV file")
    parser.add_argument("--result_store_path", type=str,
                        help="Path to store the result file")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--use_replace", action='store_true',
                        help="Whether to replace-Attack.")
    parser.add_argument("--use_insert", action='store_true',
                        help="Whether to insert-Attack.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    



    
    
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)  
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')      
    model.to(args.device)
    def CodeT5pipeline():
        
        def get_output_and_Codebleu(code,label):
            prefix = """#translate this java code to c-sharp code:
            java:"""
            input = tokenizer(prefix + code,return_tensors = "pt")
            input.to(args.device)
            output = model.generate(input.input_ids, max_length = 256)
            me = compute_metrics((output,tokenizer(label, return_tensors="pt").input_ids),tokenizer)
            return tokenizer.decode(output[0],skip_special_tokens = True),me['CodeBLEU']
        
        def cls(example):
            ret = []
            
            if type(example['code']) == list:
                for onecode in example['code']:
                    code,codebleu = get_output_and_Codebleu(onecode,example['label'])
                    ret.append({'code':code,'score':codebleu}) 
            else:
                code,codebleu = get_output_and_Codebleu(example['code'],example['label'])
                ret.append({"code":code,"score":codebleu})
            
            return ret
        
        return cls
    classifier = CodeT5pipeline()
    classifier = myclassifier(classifier)

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    #codebert_mlm.to(args.device) 


    ## Load Dataset
    eval_dataset = load_dataset("json", data_files = args.eval_data_file)
    eval_dataset = eval_dataset['train']
    substs = []
    with open(args.sub_data_file) as rf:
        for line in rf:
            item = json.loads(line.strip())
            substs.append(item["substitutes"])
    assert(len(eval_dataset) == len(substs))

   
    recoder = Recorder(args.csv_store_path)
    attacker = Attacker(args, classifier,model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe = 1, threshold_pred_score = 0)
    start_time = time.time()
    query_times = 0
    greedy_query_times = 0
    ga_query_times = 0
    for index, example in tqdm(enumerate(eval_dataset)):
        example_start_time = time.time()
        codes = example['translation']
        code = codes['java']
        true_label = codes['cs']
        subs = substs[index]
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words,Suctype,insertwords,orig_prob,current_prob = attacker.greedy_attack(code,true_label, subs,args.use_replace,args.use_insert)
        
        
        greedy_query_times = classifier.query() - query_times

        attack_type = "Greedy"
        ganb_changed_var = 0
        ganb_changed_pos = 0
        if is_success == -1 and args.use_ga:
            # 如果不成功，则使用ga_attack
            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, ganb_changed_var, ganb_changed_pos, replaced_words,Suctype = attacker.ga_attack(code,true_label, subs, initial_replace=replaced_words)
            attack_type = "GA"
            ga_query_times = classifier.query() - greedy_query_times - query_times

        example_end_time = (time.time()-example_start_time)/60
        print("Example index",index)
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time()-start_time)/60, 2), "min")
        print("Origin CodeBLEU:",orig_prob)
        print("Attacked CodeBLEU:",current_prob)
        print("Greedy_query_times: ",greedy_query_times)
        print("Ga_query_times: ",ga_query_times)
        print("ALL query times: ",query_times)
        print()
        print()
        print()
        print()
        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        recoder.write(index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, score_info, nb_changed_var, nb_changed_pos,ganb_changed_var,ganb_changed_pos,replace_info, attack_type, classifier.query() - query_times, example_end_time,Suctype,insertwords,orig_prob,current_prob)
        query_times = classifier.query()
        
        
    
    
if __name__ == '__main__':
    main()
