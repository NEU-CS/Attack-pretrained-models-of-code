import bleu
import dataflow_match
import syntax_match
import weighted_ngram_match
import numpy as np
import os



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()for label in labels]]

    return preds, labels


def compute_metrics(eval_preds,tokenzier):
    '''
    eval_preds:是模型在eval时的输出形式,是preds和labels的ids
    '''
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenzier.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenzier.pad_token_id)
    decoded_labels = tokenzier.batch_decode(labels, skip_special_tokens=True)
    
    
    hypothesis, pre_references = postprocess_text(decoded_preds, decoded_labels)

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)


    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    # calculate weighted ngram match
    
    retval = os.getcwd()
    keywords = [x.strip() for x in open(os.path.dirname(__file__) + "/keywords/c_sharp.txt", 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]
    
    result = {}
    result['BLEU'] = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)
    result['Weighted_BLEU'] = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)
    result['Syntax Match accuracy'] = syntax_match.corpus_syntax_match(references,hypothesis,"c_sharp")
    result['Dataflow_match accuracy'] = dataflow_match.corpus_dataflow_match(references,hypothesis,"c_sharp")
    result['CodeBLEU'] = 0.25*result['BLEU'] + 0.25 * result['Weighted_BLEU'] + 0.25 * result['Syntax Match accuracy'] + 0.25 * result['Dataflow_match accuracy']
    
    return result

