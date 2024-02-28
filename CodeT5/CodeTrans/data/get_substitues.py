import os
import pickle
import json
import sys
import copy
import torch
import argparse
from tqdm import tqdm

sys.path.append('../../../')
sys.path.append('../../../python_parser')

# from attacker import 
from python_parser.run_parser import get_identifiers
from utils import  get_identifier_posistions_from_code,get_masked_code_by_position2
from transformers import (RobertaForMaskedLM, RobertaTokenizer,pipeline)

import json
from utils import java_keywords

def get_substitues_by_java(file_name,store_path):
    '''
    '''
    #加载codebert-mlm模型
    basemodel = "microsoft/codebert-base-mlm"
    model = RobertaForMaskedLM.from_pretrained(basemodel)
    tokenizer = RobertaTokenizer.from_pretrained(basemodel)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer,device = 0)
    
    #读取文件
    javacodes = []
    with open(file_name,"r") as f:
        for line in f:
            javacodes.append(json.loads(line)['translation']['java'])
    
    substitutes = []
    
    for code in tqdm(javacodes):
        ids,word_tokens = get_identifiers(code,"java")
        lst = []
        for a in ids:
            lst.append(a[0])
        positions = get_identifier_posistions_from_code(word_tokens,lst)#这里得到了每个变量的mask后的代码，返回结果是一个字典
        #现在我们需要根据字典中的位置信息，将代码mask掉，然后传入pipeline中进行预测
        
        onesubstitute = {}
        
        oneres = set()
        for k,v in positions.items():
            masked_code_lst = get_masked_code_by_position2(word_tokens,v) #这里得到了mask后的代码，针对某一个变量
            masked_code = ' '.join(masked_code_lst)
            preds = fill_mask(masked_code)
            
            if len(v) > 1:
                for pred in preds:
                    for h in pred:
                        oneword = h['token_str'].strip()
                        if oneword not in java_keywords and  len(oneword) > 0:
                            oneres.add(h['token_str'].strip())
            else:
                for h in preds:
                    oneword = h['token_str'].strip()
                    if oneword not in java_keywords and  len(oneword) > 0:
                        oneres.add(h['token_str'].strip())
            onesubstitute[k] = list(oneres)

        insertdict = {}
        insertdict['code'] = code
        insertdict['substitutes'] = onesubstitute
        substitutes.append(insertdict)
    
    with open(store_path,"w") as f:
        for line in substitutes:
            f.write(json.dumps(line) + "\n")
            
if __name__ =='__main__':
    get_substitues_by_java('test_java2cs.jsonl',"test_java2cs_substitute.jsonl")