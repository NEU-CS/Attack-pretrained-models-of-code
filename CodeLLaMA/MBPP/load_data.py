'''
2023 11.17
liu jincheng
read MBPP dataset from huggingface and write it on local disk with filetype "jsonl"
'''
from datasets import load_dataset
import json

if __name__ == "__main__":
    dataset_full = load_dataset("mbpp")
    with open("mbpp.jsonl","w") as f:
        for i in dataset_full['test']:
            f.write(json.dumps(i)+"\n")