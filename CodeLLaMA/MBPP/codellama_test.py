from transformers import AutoTokenizer
import transformers
import torch
import json
from datasets import load_dataset
from tqdm import tqdm
import re

class codellama_attack:
    def __init__(self):
        self.mbpp = []
        self.mbpp_torch = []
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_mbpp_local(self):
        '''
        load mbpp dataset from local disk,return a list
        '''
        with open("mbpp.jsonl","r") as f:
            for oneline in f:
                self.mbpp.append(json.loads(oneline))
        
    def load_mbpp_online(self):
        '''
        load mbpp dataset from huggingface,return a torch.dataset
        '''
        self.mbpp_torch = load_dataset("mbpp")
        

    def load_codellama(self):
        '''
        load codellama-7b-hf from huggingface or local cache,return model,tokenzier and hugginface transformers pipeline
        '''
        self.model = "codellama/CodeLlama-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
    def mbpp_test(self):
        '''
        test codellama on mbpp dataset test split
        '''
        acc = 0
        right = 0
        all = 0
        for sample in tqdm(self.mbpp) :
            input = sample['text']
            output = self.pipeline(input,do_sample=True,
            top_k=10,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=300)[0]['generated_text']
            #匹配output中可用的部分,匹配"def到return之间的函数"
            rule = re.compile('''def[ a-zA-Z0-9_]+\([a-zA-Z0-9_\, ]*\)\:[\'\"\s\w#]*return[^\n]*''',re.M)
            func = rule.findall(output) #这里面有很多个函数，每个都测试一遍，只要有一个正确就算正确
            #将此部分作为python代码运行得到结果
            print(output)
            testlist = sample['test_list']
            f = False
            for onetest in testlist:
                for onefunc in func:
                    funcname = re.findall("(def\s*)(\w*)",onefunc)[0][1]
                    testfuncname = re.findall("(assert\s*)(\w*)",onetest)[0][1]
                    onetest.replace(testfuncname,funcname)
                    run = onefunc + "\n" + onetest
                    print("---------------")
                    print(run)
                    try: exec(run)
                    except:AssertionError
                    else: f = f | True
                if f:
                    right += 1
                all += 1
            print(right,all,sep=" ")

        acc = right/all
        print(acc)




if __name__ == "__main__":
    '''
    attack_object = codellama_attack()
    attack_object.load_mbpp_local()
    attack_object.load_codellama()
    attack_object.mbpp_test()
    '''
    s = '''def remove_char(str, ch):
        """
        This function removes the first and last occurrence of a given char from the given string
        """

        # Your code goes here

        index_first_occurrence = str.find(ch)
        index_last_occurrence = str.rfind(ch)
        if index_first_occurrence == index_last_occurrence:
            return str[:index_first_occurrence] + str[index_first_occurrence+1:]
        else:
            return str[:index_first_occurrence] + str[index_last_occurrence+1:]'''
    
    rule = re.compile('''def[ a-zA-Z0-9_]+\([a-zA-Z0-9_\, ]*\)\:[\'\"\s\w#]*return[^\n]*''',re.M)
    func = rule.findall(s)
    print(func)
    
