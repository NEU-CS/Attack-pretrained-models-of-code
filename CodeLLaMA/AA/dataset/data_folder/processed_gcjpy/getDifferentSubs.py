
'''
得到以下6种subs文件
1. 每个单词对应5个替换词
2. 每个单词对应10个替换词
3. 每个单词对应15个替换词
4. 每个单词对应20个替换词
5. 每个单词对应30个替换词
6. 每个单词对应40个替换词
ljc created in 2023 4 16
'''

import json

filepath = "valid_subs"
subsdiff = [5,7,9,11,13,15]

def get_different_subsfile():
        for num in subsdiff:
            filename = open(filepath+str(num)+'.jsonl',"w")
            with open(filepath+".jsonl") as f:
                for line in f:
                    onefile = {}
                    oneline = json.loads(line)
                    onesub = oneline["substitutes"]
                    onefile["code"] = oneline["code"]
                    onefile["substitutes"] = {}
                    for k,v in onesub.items():
                        onefile["substitutes"][k] = v[:num]
                    filename.write(json.dumps(onefile)+'\n')

if __name__ == '__main__':
    get_different_subsfile()
                

