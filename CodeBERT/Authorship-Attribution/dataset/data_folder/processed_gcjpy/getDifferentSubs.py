
'''
得到以下11种subs文件
每个单词对应5,7,9,11,13,15,20,25,30,35,40个替换词
ljc created in 2023 4 16
'''

import json

filepath = "valid_subs"
subsdiff = [5,7,9,11,13,15,20,25,30,35,40]

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
                

