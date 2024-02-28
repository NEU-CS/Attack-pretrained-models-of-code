import json
import os

def _2csv(which,direction):

    res = []

    java = []
    with open(os.path.join(which+".java-cs.txt"+".java"),"r") as f:
        for line in f:
            java.append(str(line))

    c = []
    with open(os.path.join(which+".java-cs.txt"+".cs"),"r") as f:
        for line in f:
            c.append(line)

    assert len(java) == len(c)
    with open(which + "_"+direction + ".txt", "w") as f :
        for i,v in enumerate(java):
            
            f.write(json.dumps({'id':i,'translation':{'java':v,'cs':c[i]}}) + "\n")
    

if __name__ == '__main__':
    _2csv("train","java2cs")
    _2csv("valid","java2cs")
    _2csv("test","java2cs")


