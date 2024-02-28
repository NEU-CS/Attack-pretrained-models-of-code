import sys

sys.path.append('../../../../')
sys.path.append('../../../../python_parser')

import copy
import torch
import random
from utils import select_parents, crossover, map_chromesome, mutate,_tokenize, get_identifier_posistions_from_code, get_masked_code_by_position
from run_parser import get_identifiers, get_example
from transformers import pipeline
import torch


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
        self.embedding = model_tgt.get_input_embeddings()



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
    
    def sort_sub1(self,w1:str,s1:list,w2:str,subs2:list):
        '''
        w1待替换词汇,s1成功替换后的词汇:有beam_size个
        '''
        #计算词汇的embedding,w1为已经成功替换的词汇，w2为本次将要替换的词汇
        tokenizer = self.tokenizer_tgt
        embedding = self.embedding
        w1_embedding = torch.mean(embedding(torch.tensor(tokenizer(w1)["input_ids"]).cuda()) , dim = 0)
        w2_embedding = torch.mean(embedding(torch.tensor(tokenizer(w2)["input_ids"]).cuda()) , dim = 0)
        delta_embedding = []
        s1_embedding = []
        #计算上一次成功替换后词汇的embedding,s1的大小为beam_size
        for i in range(len(s1)):
            s1_embedding.append(torch.mean(embedding(torch.tensor(tokenizer(s1[i])["input_ids"]).cuda()), dim = 0))
            delta_embedding.append(s1_embedding[i] - w1_embedding)
        subs_input_ids = tokenizer(subs2)["input_ids"]
        subs_embeddings = []
        subs_delta_embeddings = []
        #计算本次所有待替换词汇的embedding,subs2的大小为词汇表大小
        for i in range(len(subs2)):
            subs_embeddings.append(torch.mean(embedding(torch.tensor(subs_input_ids[i]).cuda()),dim = 0))
            subs_delta_embeddings.append(subs_embeddings[i] - w2_embedding)
        
        subs_delta_embeddings = torch.cat([i.unsqueeze(0) for i in subs_delta_embeddings],0)
        delta_embedding = torch.cat([i.unsqueeze(0) for i in delta_embedding],0)
        score = torch.matmul(subs_delta_embeddings,delta_embedding.T).sum(dim = 1).tolist()
        vec = list(zip(subs2,score))
        vec.sort(key = lambda x:x[1],reverse = True)
        subs_final = []
        for i in vec:
            subs_final.append(i[0])
        return subs_final
    
        

    def greedy_attack(self, code,true_label,subs,ifreplace,ifinsert,sort,accpatance):
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
        
        true_label = "LABEL_"+ str(true_label)
        results = self.classifier.predict(code)
        if type(results) == dict:
            results = [results]

        current_prob = results[0]['score']
        orig_label = results[0]['label']
        adv_code = ''
        temp_label = None

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
        before = ""
        success_sub = []
        if ifreplace:
            
            for name_and_score in sorted_list_of_names:
                tgt_word = name_and_score[0]
                all_substitues = subs[tgt_word]
                most_gap = 0.0
                candidate = None
                if sort and before and success_sub:
                    all_substitues = self.sort_sub1(before,success_sub,tgt_word,all_substitues)
                gap_list = []
                f = False
                for substitute in all_substitues:
                    if exist_words.get(substitute,False):
                        #如果当前替换的词语在之前出现过，那么不进行替换，保证语义一致性
                        continue
                    # 需要将几个位置都替换成sustitue_
                    temp_code = get_example(final_code, tgt_word, substitute, "python")
                    temp_results = self.classifier.predict(temp_code)
                    temp_label = temp_results[0]['label']
                    temp_prob = temp_results[0]['score']
                    if temp_label != orig_label:
                        # 如果label改变了，说明这个mutant攻击成功
                        is_success = 1
                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict[tgt_word])
                        candidate = substitute
                        replaced_words[tgt_word] = candidate
                        exist_words[replaced_words[tgt_word]] = True
                        exist_words[tgt_word] = False
                        adv_code = get_example(final_code, tgt_word, candidate, "python")
                        print("%s SUC! %s => %s (%.5f => %.5f)" % ('>>', tgt_word, candidate,current_prob,temp_prob), flush=True)
                        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, importance_score, nb_changed_var, nb_changed_pos, replaced_words,"replace",[],0,0
                     
                    else:
                        # 如果没有攻击成功，我们看probability的修改
                        gap = current_prob - temp_prob
                        
                        if gap > 0 and abs(gap)/abs(current_prob) > 1/accpatance :
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[tgt_word])
                            
                            candidate = substitute
                            final_code = get_example(final_code, tgt_word, candidate, "python")
                            replaced_words[tgt_word] = candidate 
                            exist_words[candidate] = True
                            exist_words[tgt_word] = False
                            before = tgt_word
                            success_sub = [candidate]
                            print("%s ACC! %s => %s (%.5f => %.5f)" % \
                                    ('>>', tgt_word, candidate,
                                    current_prob,
                                    temp_prob), flush=True)
                            current_prob = temp_prob
                            f = True
                            break
                        else:
                            gap = current_prob - temp_prob
                            gap_list.append((gap,substitute))
                
                if not f:
                    gap_list = sorted(gap_list,key = lambda x:x[0],reverse=True)
                    most_gap = gap_list[0][0]
                    candidate = gap_list[0][1]
                    if most_gap > 0:

                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict[tgt_word])
                        current_prob = current_prob - most_gap
                        final_code = get_example(final_code, tgt_word, candidate, "python")
                        replaced_words[tgt_word] = candidate
                        exist_words[candidate] = True
                        print("%s FINALACC! %s => %s (%.5f => %.5f)" % \
                            ('>>', tgt_word, candidate,
                            current_prob + most_gap,
                            current_prob), flush=True)
                        exist_words[tgt_word] = False
                            
                        
                    '''
                    这里改成模拟退火
                    print("%s NOACC! %s => %s (%.5f => %.5f)" % \
                        ('>>', tgt_word, candidate,
                        current_prob + most_gap,
                        current_prob), flush=True)
                    temp_p = random.random()       
                    if temp_p > p:
                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict[tgt_word])
                        current_prob = current_prob - most_gap
                        final_code = get_example(final_code, tgt_word, candidate, "python")
                        replaced_words[tgt_word] = candidate
                        exist_words[replaced_words[tgt_word]] = True
                        print("%s SNT! %s => %s (%.5f => %.5f)" % \
                        ('>>', tgt_word, candidate,
                        current_prob + most_gap,
                        current_prob), flush=True)
                    else:
                        print(">> NOSNT!",flush=True)
                        replaced_words[tgt_word] = tgt_word
                        exist_words[replaced_words[tgt_word]] = True
                    '''
                adv_code = final_code
        # 添加死代码，死代码为变量的定义和print(变量名)的操作
        insert_words = []
        if ifinsert:
            vocab = list(names_positions_dict.keys())
            for k,v in subs.items():
                vocab += v

            re = [(";"+i+" = 0",";print("+i+")") for i in vocab]
            vocab = []
            for i in re:
                vocab.append(i[0]);vocab.append(i[1])
            flag = False
            
            for addcode in vocab:
                #只有当变量声明语句被插入了才会插入print变量的语句
                
                if addcode.startswith(";print("):
                    if flag == False:
                        continue
                else:
                    flag = False
                
                
                temp_code = final_code + addcode

                temp_result = self.classifier.predict(temp_code)
                if temp_result[0]['label'] != true_label:
                    final_code = temp_code
                    adv_code = final_code
                    insert_words.append(addcode)
                    print(">> SUC! INSERT %s"%(addcode),flush=True)
                    is_success = 1
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, importance_score, nb_changed_var, nb_changed_pos, replaced_words,"insert",insert_words,0,0
                else:
                    nowprob = temp_result[0]['score']
                    if nowprob < current_prob:
                        flag = True
                        final_code = temp_code
                        insert_words.append(addcode)
                        print(">> ACC! INSERT %s %.5f=>%.5f"%(
                            addcode,current_prob,nowprob
                        ),flush=True)
                        current_prob = nowprob
                    else:
                        pass
                        """
                        print(">> NOACC! INSERT %s %.5f=>%.5f CHECK SNT"%(
                            addcode,current_prob,nowprob
                        ),flush=True)
                        temp_p = random.random()
                        if temp_p > p :
                            final_code = temp_code
                            print(">> SNT! INSERT %s %.5f=>%.5f"%(
                                addcode,current_prob,nowprob
                            ),flush=True)
                            current_prob = nowprob
                        else:
                            print(">> NOSNT",flush=True)
                        """
                        
                adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, importance_score, nb_changed_var, nb_changed_pos, replaced_words,"failed",insert_words,0,0
    
    
            
