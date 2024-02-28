import sys

sys.path.append('../../../../')
sys.path.append('../../../../python_parser')

import copy
import torch
import random
from utils import select_parents, crossover, map_chromesome, mutate,_tokenize, get_identifier_posistions_from_code, get_masked_code_by_position
from run_parser import get_identifiers, get_example
from transformers import pipeline
import re

def compute_fitness(chromesome,classifier, orig_prob ,code):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "python")
    result = classifier.predict(temp_code)
    # 计算fitness function
    fitness_value = orig_prob - result[0]['score']
    return fitness_value, result[0]['label']



def get_importance_score(words_list: list,variable_names: list, classifier,orig_prob):
    '''Compute the importance score of each variable'''
    # 1. 过滤掉所有的keywords.
    
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = {}

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.


    for variable, masked_one_variable in masked_token_list.items():
        new_code = ' '.join(masked_one_variable)
        new_example[variable] = new_code

    
    results = {}
    for variable,code_masked in new_example.items():
        results[variable] = classifier.predict(code_masked)[0]['score']

    #orig_prob = results[0]['score'] #这个为什么不直接参数传进来，非要重新计算一次

    importance_score = {}

    for variable,prob in results.items():
        importance_score[variable] = orig_prob - prob
    return importance_score, replace_token_positions, positions

class Attacker():
    def __init__(self, args, classifier, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.classifier = classifier
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score


    def generate_deadcode(self,var_name:str,var_list:list):
        
        ret = []
        ret.append("print(" + var_name + ")")
        same = var_name + "=" + var_list[random.randint(0,len(var_list)-1)][0]
        st = """if False:\n\t"""
        st += same
        ret.append(st)
        st = "while False:\n\t"
        st += same
        ret.append(st)     
        return ret
    
    def get_program_stmt_nums(self,code:str):
        '''
        获得一段代码所有的可以插入的位置
        这里有问题，没有考虑;和\运算符，后面还需要改
        '''
        return code.split("\n")
    
    

    def ga_attack(self, code , true_label , subs , initial_replace=None):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
        # 先得到tgt_model针对原始Example的预测信息.
        results = self.classifier.predict(code)
        if type(results) == dict:
            results = [results]
        
        current_prob = results[0]['score']
        orig_label = results[0]['label']
        adv_code = ''
        temp_label = None

        identifiers, code_tokens = get_identifiers(code, 'python')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)
        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..


        variable_names = list(subs.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, None
            
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, None

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1

        # 我们可以先生成所有的substitues
        variable_substitue_dict = {}

        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = subs[tgt_word]

        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        # 关于chromesome的定义: {tgt_word: candidate, tgt_word_2: candidate_2, ...}
        for tgt_word in variable_substitue_dict.keys():
            # 这里进行初始化
            if initial_replace is None:
                # 对于每个variable: 选择"影响最大"的substitues
                replace_examples = []
                substitute_list = []

                most_gap = 0.0
                initial_candidate = tgt_word
                tgt_positions = names_positions_dict[tgt_word]
                
                # 原来是随机选择的，现在要找到改变最大的.
                for a_substitue in variable_substitue_dict[tgt_word]:
                    # a_substitue = a_substitue.strip()
                    
                    substitute_list.append(a_substitue)
                    # 记录下这次换的是哪个substitue
                    temp_code = get_example(code, tgt_word, a_substitue, "python") 
                    replace_examples.append(temp_code)

                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    continue
                
                results = self.classifier.predict(replace_examples)

                _the_best_candidate = -1
                for index, oneresult in enumerate(results):
                    temp_label = oneresult['label']
                    gap = current_prob - oneresult['score']
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = compute_fitness(temp_chromesome, self.classifier,current_prob,code)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)
        # 这里的超参数还是的调试一下.

        for i in range(max_iter):
            _temp_mutants = []
            for j in range(self.args.eval_batch_size):
                p = random.random()
                chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
                if p < cross_probability: # 进行crossover
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, child_2 = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else: # 进行mutates
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)
            
            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _temp_code = map_chromesome(mutant, code, "python")
                feature_list.append(_temp_code)
            if len(feature_list) == 0:
                continue
            results = self.classifier.predict(feature_list)
            mutate_fitness_values = []
            for index, oneresult in enumerate(results):
                if oneresult['label'] != orig_label:
                    adv_code = map_chromesome(_temp_mutants[index], code, "python")
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])
                    print("GA SUC!",flush = True)
                    return code, prog_length, adv_code, true_label, orig_label, oneresult['label'], 1, variable_names, None, nb_changed_var, nb_changed_pos, _temp_mutants[index],"GA"
                _tmp_fitness = current_prob - oneresult['score']
                mutate_fitness_values.append(_tmp_fitness)
            
            # 现在进行替换.
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    # 替换.
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None,"failed"
        

    def insert_attack(self,pos:int,source_code:str,var_tgt,candidate,insert_deadcode:str):
            all_insert_pos = self.get_program_stmt_nums(source_code)
            code = all_insert_pos[:pos] + [insert_deadcode] + all_insert_pos[:pos]
            code = "".join(code)
            temp_result = self.classifier.predict(code)
            return temp_result
    
    def repalce_attack(self,pos,source_code:str,var_tgt:str,candidate:str,insert_deadcode):
            code = get_example(source_code,var_tgt,candidate,"python")
            temp_result = self.classifier.predict(code)
            return temp_result
    

        
    
    def greedy_attack(self, code,true_label,subs,ifreplace,ifinsert):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
            # 先得到tgt_model针对原始Example的预测信息.
        #理想状态：把它写成一个框架，可以在任意一点随机调用其他各种attack方式
        if not ifreplace and not ifinsert:
            raise "Need at least one semantic-preserved transformation way"
        
        
        true_label = "LABEL_"+ str(true_label)
        results = self.classifier.predict(code)
        if type(results) == dict:
            results = [results]

        current_prob = results[0]['score']
        orig_label = results[0]['label']
        adv_code = ''
        temp_label = None
        print(orig_label,true_label,current_prob)
        p = 0.75
        identifiers, code_tokens = get_identifiers(code, 'python')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)
        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..


        variable_names = list(subs.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None,None,None,0,0
            
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None,None,None,0,0


        

        # 计算importance_score.

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(
                                                words,
                                                variable_names,
                                                self.classifier,
                                                current_prob)

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None,None,None,0,0


        """_summary_

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score,将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置:token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]
            
            names_to_importance_score[name] = total_score
        """
        
        
        sorted_list_of_names = sorted(importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_code = copy.deepcopy(code)
        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}
        exist_words = {} #记录已经存在过的变量名，变量名一定不能重复

        replace_index = 0
        iter_nums = len(sorted_list_of_names)
        trans_numbers = 0
        trans = []
        insert_words = []
        insert_pos = 2
        if ifinsert:
            trans_numbers += 1
            trans.append("insert")
        if ifreplace:
            trans_numbers += 1
            trans.append("replace")
        
        for iter in range(iter_nums):
            trans_index = random.randint(0,trans_numbers - 1)
            
            transf = trans[trans_index]
            if transf == "replace":
                tgt_word = sorted_list_of_names[replace_index][0]
                all_subs = subs[tgt_word]
                temp_inputs = []
                for onesub in all_subs:
                    if exist_words.get(onesub,False) == False:
                        onecode = get_example(final_code,tgt_word,onesub,"python")
                        temp_inputs.append(onecode)
                temp_results = self.classifier.predict(temp_inputs)
                lowest_index = 0
                lowest_prob = temp_results[0]['score']
                for i,v in enumerate(temp_results):
                    if v['label'] != true_label:
                        print("SUC!",tgt_word,"=>",all_subs[i],"prob:",current_prob,"=>",v['score'],"label",v['label'],sep=" ")
                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict[tgt_word])
                        return code,prog_length,temp_inputs[i],true_label,orig_label,v['label'],1,variable_names,importance_score, nb_changed_var, nb_changed_pos, replaced_words,"replace",insert_words,0,0
                    if v['score'] < lowest_prob:
                        lowest_prob = v['score']
                        lowest_index = i
                
                if lowest_prob < current_prob:
                    print("ACC!",tgt_word,"=>",all_subs[lowest_index],"prob:",current_prob,"=>",lowest_prob,sep=" ")
                    current_prob = lowest_prob
                    final_code = temp_inputs[lowest_index]
                    exist_words[all_subs[lowest_index]] = True
                    exist_words[tgt_word] = False
                    replaced_words[tgt_word] = all_subs[lowest_index]
                else:
                    print("NOACC!",tgt_word,"=>",all_subs[lowest_index],"prob:",current_prob,"=>",lowest_prob,sep=" ")
                
                replace_index += 1
                
            elif transf =="insert":
                code_lst = self.get_program_stmt_nums(final_code)
                ids , _ = get_identifiers("".join(code_lst[:insert_pos]),"python")
                insert_pos += 1
                if(len(ids) >= 1):
                    p = random.randint(0,len(ids)-1)
                    deadcodes = self.generate_deadcode(ids[p][0],ids)
                    for onedeadcode in deadcodes:
                        temp_results = self.insert_attack(insert_pos,final_code,"","",onedeadcode)
                        label = temp_results[0]['label']
                        prob = temp_results[0]['score']
                        if label != true_label:
                            print("SUC!","insert",onedeadcode,"prob:",current_prob,"=>",prob,"label",label,sep=" ")
                            insert_words.append(onedeadcode)
                            return code,prog_length,"".join(code_lst[insert_pos:] + [onedeadcode] + code_lst[:insert_pos]),true_label,orig_label,label,1,variable_names,importance_score, nb_changed_var, nb_changed_pos, replaced_words,"insert",insert_words,0,0
                        else:
                            if prob < current_prob:
                                print("ACC!","insert",onedeadcode,"prob:",current_prob,"=>",prob,sep=" ")
                                current_prob = prob
                                insert_words.append(onedeadcode)
                                final_code = "".join(code_lst[insert_pos:] + [onedeadcode] + code_lst[:insert_pos])
                            else:
                                print("NOACC!","insert",onedeadcode,"prob:",current_prob,"=>",prob,sep=" ")
                        
                
            

                
                        
        
            

        return code, prog_length, final_code, true_label, orig_label, temp_label, is_success, variable_names, importance_score, nb_changed_var, nb_changed_pos, replaced_words,"failed",insert_words,0,0
    
    
            
