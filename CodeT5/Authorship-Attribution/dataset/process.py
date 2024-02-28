import os
import sys
sys.path.append('../../../')
sys.path.append('../../../python_parser')
from run_parser import get_identifiers, get_code_tokens
from parser_folder import remove_comments_and_docstrings
import random
import pandas

def preprocess_gcjpy(split_portion):
    '''
    预处理文件.
    需要将结果分成train和valid
    '''
    data_name = "gcjpy"
    folder = os.path.join('./data_folder', data_name)
    output_dir = os.path.join('./data_folder', "processed_" + data_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    authors = os.listdir(folder)

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for index, name in enumerate(authors):
            f.write(str(index) + '\t' + name + '\n')


    train_example = []
    valid_example = []
    for index, name in enumerate(authors):
        files = os.listdir(os.path.join(folder, name))
        tmp_example = []
        for file_name in files:
            with open(os.path.join(folder, name, file_name)) as code_file:
                lines_after_removal = []
                for a_line in code_file.readlines():
                    if  a_line.strip().startswith("import") or a_line.strip().startswith("#") or a_line.strip().startswith("from"):
                        continue
                    lines_after_removal.append(a_line)
                
                content = "".join(lines_after_removal)
                print(content)
                code_tokens = get_code_tokens(content, 'python')
                content = " ".join(code_tokens)
                new_content = content + ' <CODESPLIT> ' + str(index) + '\n'
                tmp_example.append(new_content)
        split_pos = int(len(tmp_example) * split_portion)
        train_example += tmp_example[0:split_pos]
        valid_example += tmp_example[split_pos:]

            # 8 for train and 2 for validation

    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        for example in train_example:
            f.write(example)
    
    with open(os.path.join(output_dir, "valid.txt"), 'w') as f:
        for example in valid_example:
            f.write(example)

def preprocess_java40(split_portion = 0.8):
    '''
    预处理文件.
    需要将结果分成train和valid
    '''
    data_name = "java40"
    folder = os.path.join('./data_folder', data_name)
    output_dir = os.path.join('./data_folder', "processed_" + data_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    authors = os.listdir(folder)

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for index, name in enumerate(authors):
            if name[0] == '.':
                continue
            f.write(str(index) + '\t' + name + '\n')



    train_example = []
    valid_example = []
    for index, name in enumerate(authors):
        if name[0] == '.':
            continue
        repos = os.listdir(os.path.join(folder, name))
        for repo in repos:
            files = os.listdir(os.path.join(folder, name, repo))
            tmp_example = []
            for file_name in files:
                with open(os.path.join(folder, name, repo, file_name), encoding="utf8", errors='ignore') as code_file:
                    lines_after_removal = []
                    for a_line in code_file.readlines():
                        if a_line.startswith("package") or a_line.startswith("import"):
                            continue
                        lines_after_removal.append(a_line)

                    content = "\n".join(lines_after_removal)
                    identifiers, code_tokens = get_identifiers(content, 'java')
                    content = " ".join(code_tokens)
                    new_content = content + ' <CODESPLIT> ' + str(index) + '\n'
                    tmp_example.append(new_content)
            split_pos = int(len(tmp_example) * split_portion)
            train_example += tmp_example[0:split_pos]
            valid_example += tmp_example[split_pos:]

            # 8 for train and 2 for validation


    random.shuffle(train_example)
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        for example in train_example:
            f.write(example)
    

    random.shuffle(valid_example)
    with open(os.path.join(output_dir, "valid.txt"), 'w') as f:
        for example in valid_example:
            f.write(example)


def transformTxt2Csv(filename):
    folder = os.path.join('./data_folder/processed_gcjpy', filename+".txt")
    dataframe = pandas.DataFrame(columns=["prompt","completion"])
    rownum = 0
    with open(folder,"r") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            code = line.split(" <CODESPLIT> ")[0]
            code = code.replace("\\n", "\n").replace('\"','"')
            code = code + " <CODESPLIT> "
            label = line.split(" <CODESPLIT> ")[1]
            dataframe.loc[rownum] = [code,label]
            rownum += 1
    dataframe.to_csv(filename+".csv",index = 0)

if __name__ == "__main__":
    preprocess_gcjpy(0.8)
    transformTxt2Csv("valid")