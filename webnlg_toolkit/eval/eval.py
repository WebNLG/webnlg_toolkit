__author__='thiagocastroferreira,liamcripwell'

"""
Author: Organizers of the 3rd WebNLG Challenge
Date: 21/02/2023
Description:
    This script aims to evaluate the output of data-to-text NLG models by computing 
    popular automatic metrics such as BLEU, METEOR, chrF++, TER and BERT-Score.
    
    ARGS:
        usage: eval.py [-h] -R REFERENCE -H HYPOTHESIS [-lng LANGUAGE] [-nr NUM_REFS]
               [-m METRICS] [-nc NCORDER] [-nw NWORDER] [-b BETA]

        optional arguments:
          -h, --help            show this help message and exit
          -R REFERENCE, --reference REFERENCE
                                reference translation
          -H HYPOTHESIS, --hypothesis HYPOTHESIS
                                hypothesis translation
          -lng LANGUAGE, --language LANGUAGE
                                evaluated language
          -nr NUM_REFS, --num_refs NUM_REFS
                                number of references
          -m METRICS, --metrics METRICS
                                evaluation metrics to be computed
          -nc NCORDER, --ncorder NCORDER
                                chrF metric: character n-gram order (default=6)
          -nw NWORDER, --nworder NWORDER
                                chrF metric: word n-gram order (default=2)
          -b BETA, --beta BETA  chrF metric: beta parameter (default=2)

    EXAMPLE:
        ENGLISH: 
            python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt
        RUSSIAN:
            python3 eval.py -R data/ru/reference -H data/ru/hypothesis -lng ru -nr 1 -m bleu,meteor,chrf++,ter,bert
"""

import os
import gc
import sys
import nltk
import copy
import pyter
import torch
import codecs
import logging
import argparse
import subprocess

import pandas as pd
from tqdm import tqdm
from razdel import tokenize
from bert_score import score
from tabulate import tabulate
from sacrebleu.metrics import BLEU
from parent import parent

from webnlg_toolkit.eval.metrics.chrF import computeChrF
from webnlg_toolkit.eval.metrics.bleurt.bleurt import score as bleurt_score
from webnlg_toolkit.eval.metrics.SEScore2.SEScore2 import *

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


METEOR_PATH = 'webnlg_toolkit/eval/metrics/meteor-1.5/meteor-1.5.jar'
SESCORE2_PATH = 'webnlg_toolkit/eval/metrics/SEScore2'

BERT_LANGS = ["br", "cy", "en", "ga", "ru"]
METEOR_LANGS = ["en", "ru"]


def parse(refs_path, hyps_path, graph_path, num_refs, lng='en'):
    logging.info('STARTING TO PARSE INPUTS...')
    print('STARTING TO PARSE INPUTS...')
    # references
    references = []
    if isinstance(refs_path, list):
        # handle case where actual DS of texts are provided
        references = [eval(y) if isinstance(y, str) else y for y in refs_path]
    elif refs_path.endswith('.csv'):
        references = [eval(y) if isinstance(y, str) else y for y in pd.read_csv(refs_path)['ref']]
        for i in range(len(references)):
            references[i] += ["" for _ in range(num_refs - len(references[i]))]
    else:
        for i in range(num_refs):
            fname = refs_path + str(i) if num_refs > 1 else refs_path
            with codecs.open(fname, 'r', 'utf-8') as f:
                #texts = f.read().split('\n')
                texts = [x.strip() for x in f.readlines()]
                for j, text in enumerate(texts):
                    if len(references) <= j:
                        references.append([text])
                    else:
                        references[j].append(text)

    # references tokenized
    references_tok = copy.copy(references)
    for i, refs in enumerate(references_tok):
        if lng == 'ru':
            references_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
        else:
            references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

    # hypothesis
    if isinstance(hyps_path, list):
        # handle case where actual DS of texts are provided
        hypothesis = hyps_path
    elif refs_path.endswith('.csv'):
        hypothesis = [x.strip() for x in pd.read_csv(refs_path)['output']]
    else:
        with codecs.open(hyps_path, 'r', 'utf-8') as f:
            #hypothesis = f.read().split('\n')
            hypothesis = [x.strip() for x in f.readlines()]

    # hypothesis tokenized
    hypothesis_tok = copy.copy(hypothesis)
    if lng == 'ru':
        hypothesis_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in hypothesis_tok]
    else:
        hypothesis_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in hypothesis_tok]

    # raw graph
    if isinstance(graph_path, list):
        pass
    else:
        with codecs.open(graph_path, 'r', 'utf-8') as f:
            graphs = [x.strip() for x in f.readlines()]


    logging.info('FINISHING TO PARSE INPUTS...')
    print('FINISHING TO PARSE INPUTS...')
    return references, references_tok, hypothesis, hypothesis_tok, graphs


def sacrebleu_score(references, hypothesis, num_refs):
    refs = []
    for i in range(num_refs):
        # allow for variable number of references per example
        refs_ = [ref[i] if len(ref) > i and ref[i].strip() != "" else None for ref in references]
        refs.append(refs_)

    bleu = BLEU(effective_order=True)
    bleus = [bleu.sentence_score(hypothesis[i], [r[i] for r in refs]).score for i in range(len(hypothesis))]

    return bleus


def meteor_score(references, hypothesis, num_refs, lng='en'):
    logging.info('STARTING TO COMPUTE METEOR...')
    print('STARTING TO COMPUTE METEOR...')
    hyps_tmp, refs_tmp = 'hypothesis_meteor', 'reference_meteor'

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis))

    linear_references = []
    for refs in references:
        for i in range(num_refs):
            linear_references.append(refs[i])

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    try:
        command = 'java -Xmx2G -jar {0} '.format(METEOR_PATH)

        # handle unsupported langs
        if lng not in METEOR_LANGS:
            print(f"Language {lng} not officially supported by METEOR. Reverting to language-independent version.")
            lng = "other"

        if lng == "other":
            command += '{0} {1} -l {2} -r {3}'.format(hyps_tmp, refs_tmp, lng, num_refs)
        else:
            command += '{0} {1} -l {2} -norm -r {3}'.format(hyps_tmp, refs_tmp, lng, num_refs)
        print(f"METEOR command executed: {command}")
        result = subprocess.check_output(command, shell=True)
        meteor = result.split(b'\n')[-2].split()[-1]
    except:
        logging.error('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
        print('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
        meteor = -1

    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info('FINISHING TO COMPUTE METEOR...')
    print('FINISHING TO COMPUTE METEOR...')
    return float(meteor)


def chrF_score(references, hypothesis, num_refs, nworder, ncorder, beta):
    logging.info('STARTING TO COMPUTE CHRF++...')
    print('STARTING TO COMPUTE CHRF++...')
    hyps_tmp, refs_tmp = 'hypothesis_chrF', 'reference_chrF'

    # check for empty lists
    references_, hypothesis_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append(refs_)
            hypothesis_.append(hypothesis[i])

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis_))

    linear_references = []
    for refs in references_:
        linear_references.append('*#'.join(refs[:num_refs]))

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    rtxt = codecs.open(refs_tmp, 'r', 'utf-8')
    htxt = codecs.open(hyps_tmp, 'r', 'utf-8')

    try:
        totalF, averageTotalF, totalPrec, totalRec = computeChrF(rtxt, htxt, nworder, ncorder, beta, None)
    except:
        logging.error('ERROR ON COMPUTING CHRF++.')
        print('ERROR ON COMPUTING CHRF++.')
        totalF, averageTotalF, totalPrec, totalRec = -1, -1, -1, -1
    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info('FINISHING TO COMPUTE CHRF++...')
    print('FINISHING TO COMPUTE CHRF++...')
    return totalF, averageTotalF, totalPrec, totalRec


def ter_score(references, hypothesis, num_refs):
    logging.info('STARTING TO COMPUTE TER...')
    print('STARTING TO COMPUTE TER...')
    ter_scores = []
    for hyp, refs in zip(hypothesis, references):
        candidates = []
        for ref in refs[:num_refs]:
            if len(ref) == 0:
                ter_score = 1
            else:
                try:
                    ter_score = pyter.ter(hyp.split(), ref.split())
                except:
                    ter_score = 1
            candidates.append(ter_score)

        ter_scores.append(min(candidates))

    logging.info('FINISHING TO COMPUTE TER...')
    print('FINISHING TO COMPUTE TER...')
    return ter_scores


def bert_score_(references, hypothesis, lng='en'):
    logging.info('STARTING TO COMPUTE BERT SCORE...')
    print('STARTING TO COMPUTE BERT SCORE...')
    for i, refs in enumerate(references):
        references[i] = [ref for ref in refs if ref.strip() != '']

    try:
        if lng not in BERT_LANGS:
            print(f"Language {lng} not officially supported by BERT Score metric.")
        P, R, F1 = score(hypothesis, references, lang=lng)
        logging.info('FINISHING TO COMPUTE BERT SCORE...')
    #     print('FINISHING TO COMPUTE BERT SCORE...')
        P, R, F1 = list(P), list(R), list(F1)
    except:
        print("BERTScore calculation failed... setting to default value of 0.")
        P, R, F1 = 0, 0, 0
    return P, R, F1

def bleurt(references, hypothesis, num_refs, checkpoint = "webnlg_toolkit/eval/metrics/bleurt/BLEURT-20"):
    refs, cands = [], []
    for i, hyp in enumerate(hypothesis):
        for ref in references[i][:num_refs]:
            cands.append(hyp)
            refs.append(ref)

    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(references=refs, candidates=cands)
    scores = [max(scores[i:i+num_refs]) for i in range(0, len(scores), num_refs)]

    # Unload model
    del scorer
    gc.collect()
    torch.cuda.empty_cache()

    return round(sum(scores) / len(scores), 3)

def parent_score(references, hypothesis, graphs):
    def _table(table):
        """Convert table to field, value format."""
        def _tokenize(x):
            return nltk.word_tokenize(" ".join(x.lower().split("_")))
        # Parse triple
        graph = [triple.split(" | ") for triple in eval(table)]
        return [[relation, _tokenize(head) + _tokenize(value)]
                for (head, relation, value) in graph]
    
    def _text(x):
        """Lowercase and tokenize text."""
        return nltk.word_tokenize(x.lower())

    predictions = [_text(pred) for pred in hypothesis]
    references = [[_text(ref) for ref in refs] for refs in references]
    graphs = [_table(table) for table in graphs]
    
    # Compute parent score on system level
    _, _, f_score = parent(
        predictions,
        references,
        graphs,
        avg_results=True,
        n_jobs=32,
        use_tqdm=True
    )

    torch.cuda.empty_cache()
    return f_score

def eredat(hypothesis, graphs):
    def _text(text):
        return " ".join(nltk.word_tokenize(text)).strip()
    
    def _table_linearize(table):
        '''
        Linearize table data into one string
        '''
        graph = [triple.split(" | ") for triple in eval(table)]
        linearized_triple = ""
        for triple in graph:
            subject, predicate, object = triple
            linearized_triple += " [S] " + subject + " [P] " + predicate + " [O] " + object
        return linearized_triple.replace("_", " ").strip()
    
    def encode_batched(model, sentences):
        return model.encode(sentences)
        
    def compute_cosine(embed_pred, embed_graph):
        import numpy as np
        # Normalize the vectors to unit vectors (divide each vector by its norm)
        norm_embed_pred = embed_pred / np.linalg.norm(embed_pred, axis=1, keepdims=True)
        norm_embed_graph = embed_graph / np.linalg.norm(embed_graph, axis=1, keepdims=True)

        # Compute the cosine similarity only for i=j
        cosine_scores = np.sum(norm_embed_pred * norm_embed_graph, axis=1)
        return cosine_scores
    
    # Load model
    model = SentenceTransformer('teven/bi_all_bs192_hardneg_finetuned_WebNLG2017')

    predictions = [_text(pred) for pred in hypothesis]
    graphs = [_table_linearize(table) for table in graphs]
    assert len(predictions) == len(graphs)

    # Compute embeddings
    embed_pred = encode_batched(model, predictions)
    embed_graph = encode_batched(model, graphs)

    # Compute similarity
    cosine_scores = compute_cosine(embed_pred, embed_graph)

    # Unload model
    del model, embed_graph, embed_pred, predictions, graphs
    gc.collect()
    torch.cuda.empty_cache()

    return cosine_scores.mean()

def factspotter(hypothesis, graphs):
    def _text(text):
        return text.strip()

    def _table_linearize(table):
        '''
        Linearize table data into one string
        '''
        graph = [triple.split(" | ") for triple in eval(table)]
        linearized_triples = []
        for triple in graph:
            subject, predicate, object = triple
            triple_to_add = subject + ", " + predicate + ", " + object
            linearized_triples.append(triple_to_add.replace("_", " ").strip())
        return linearized_triples

    def sentence_cls_score(input_strings, predicate_cls_model, predicate_cls_tokenizer):
        tokenized_cls_input = predicate_cls_tokenizer(input_strings, truncation=True, padding=True,
                                                    return_token_type_ids=True)
        input_ids = torch.Tensor(tokenized_cls_input['input_ids']).long().to(torch.device("cuda"))
        token_type_ids = torch.Tensor(tokenized_cls_input['token_type_ids']).long().to(torch.device("cuda"))
        attention_mask = torch.Tensor(tokenized_cls_input['attention_mask']).long().to(torch.device("cuda"))
        prev_cls_output = predicate_cls_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        softmax_cls_output = torch.softmax(prev_cls_output.logits, dim=1, )
        return softmax_cls_output

    def mapped_score_in_batch(gt_cls_input, gt_count, model, tokenizer, batch_size=8):
        # Calculate GT Entailment score in batch
        torch.cuda.empty_cache()
        batch_size = batch_size
        # get cls score for each batch of GT
        batched_gt_cls = [gt_cls_input[i:i + batch_size] for i in range(0, len(gt_cls_input), batch_size)]
        # cls_gt = []
        cls_entail_score = []
        for golden_batch in tqdm(batched_gt_cls, 'GT CLS Progress'):
            tmp_cls = sentence_cls_score(golden_batch, model, tokenizer)
            cls_entail_score.extend([float(x[0]) for x in tmp_cls])

        # Map the count number into the scores
        index = 0
        avg_mapped_score = []
        for nb in gt_count:
            avg = sum(cls_entail_score[index:index+nb]) / nb
            avg_mapped_score.append(avg)
            index += nb

        assert len(avg_mapped_score) == len(gt_count)

        # factSpotter_Score = sum(avg_mapped_score) / len(avg_mapped_score)
        return avg_mapped_score
    
    def make_pairs(preds, graphs):
        pairs = []
        graph_size = []
        for i, pred in enumerate(preds):
            pairs.extend([(pred, triple) for triple in graphs[i]])
            graph_size.append(len(graphs[i]))
        return pairs, graph_size

    # Processe pred and graph
    predictions = [_text(pred) for pred in hypothesis]
    graphs = [_table_linearize(table) for table in graphs]
    assert len(predictions) == len(graphs)

    # Combine pred and graph as pairs
    pairs, graph_size = make_pairs(predictions, graphs)

    # Load Models
    tokenizer = AutoTokenizer.from_pretrained("Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
    model = AutoModelForSequenceClassification.from_pretrained("Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
    model.to(torch.device("cuda"))

    # Compute FactSpotter score
    factscore = mapped_score_in_batch(pairs, graph_size, model, tokenizer, 8)

    # Unload model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return sum(factscore) / len(factscore)

def sescore2(references, hypothesis, num_refs, batch_size=16, path=SESCORE2_PATH):
    # Add SEScocre2 directory into the working repo
    sys.path.append(path)

    def _text(text):
        return " ".join(nltk.word_tokenize(text)).strip()

    # Load SEScore2
    scorer = SEScore2('en', mode="pretrained")

    # Extract pred and ref
    predictions = [_text(pred) for pred in hypothesis]

    # Expand predictions list to match the number of non-empty references
    expanded_predictions = []
    actual_flat_references = []
    for i, refs in enumerate(references):
        # Filter out empty references
        non_empty_refs = [ref for ref in refs if ref.strip()]
        
        # Expand prediction based on the actual number of non-empty references
        expanded_predictions.extend([predictions[i]] * len(non_empty_refs))
        
        # Flatten non-empty references
        actual_flat_references.extend([_text(ref) for ref in non_empty_refs])

    
    # Check if expanded_predictions and flat_references lengths match
    assert len(expanded_predictions) == len(actual_flat_references), \
        "Expanded predictions and flattened references must have the same length."
    
    # Compute sescore2 score
    sescore = scorer.score(actual_flat_references, expanded_predictions, batch_size=batch_size)

    # Keep the max score for each prediction based on the number of non-empty references
    scores = []
    start_idx = 0
    for refs in references:
        non_empty_count = len([ref for ref in refs if ref.strip()])
        if non_empty_count > 0:
            end_idx = start_idx + non_empty_count
            scores.append(max(sescore[start_idx:end_idx]))
            start_idx = end_idx

    assert len(scores) == len(predictions), "Number of scores must match number of predictions."

    # Unload model
    del sescore
    gc.collect()
    torch.cuda.empty_cache()

    return sum(scores) / len(scores)

def DQE(hypothesis, graphs, path='webnlg_toolkit/eval/metrics/DQE'):
    # Add DQE directory into the working repo
    sys.path.append(path)
    from questeval.questeval_metric import QuestEval

    def _text(text):
        return " ".join(nltk.word_tokenize(text)).strip()

    def _table_linearize(table):
        '''
        Linearize table data into one string
        '''
        graph = [triple.split(" | ") for triple in eval(table)]
        linearized_triples = []
        for triple in graph:
            subject, predicate, object = triple
            linearized_triples.append((subject + " | " + predicate + " | " + object).replace("_", " ").strip())
        return linearized_triples
    
    # Load Model
    questeval = QuestEval(task="data2text", no_cuda=False)
    
    # Processe pred and graph
    predictions = [_text(pred) for pred in hypothesis]
    graphs = [_table_linearize(table) for table in graphs]
    assert len(predictions) == len(graphs)

    # Compute DQE score
    scores = questeval.corpus_questeval(
                hypothesis=predictions, 
                sources=graphs
            )

    # Unload model
    del questeval
    gc.collect()
    torch.cuda.empty_cache()
    
    return scores['corpus_score']


def run(refs_path, hyps_path, graph_path, num_refs, lng='en', metrics='bleu,meteor,chrf++,ter,bert,bleurt,eredat,factspotter,parent,dqe,sescore', ncorder=6, nworder=2, beta=2):
    metrics = metrics.lower().split(',')
    references, references_tok, hypothesis, hypothesis_tok, graphs = parse(refs_path, hyps_path, graph_path, num_refs, lng)

    result = {}
    
    logging.info('STARTING EVALUATION...')
    if 'bleu' in metrics:
        bleus = sacrebleu_score(references, hypothesis, num_refs)
        result["bleu"] = bleus
    if 'meteor' in metrics:
        meteor = meteor_score(references_tok, hypothesis_tok, num_refs, lng=lng)
        result['meteor'] = meteor
    if 'chrf++' in metrics:
        chrf, _, _, _ = chrF_score(references, hypothesis, num_refs, nworder, ncorder, beta)
        result['chrf++'] = chrf
    if 'ter' in metrics:
        ters = ter_score(references_tok, hypothesis_tok, num_refs)
        result['ter'] = ters
    if 'bert' in metrics:
        P, R, F1 = bert_score_(references, hypothesis, lng=lng)
        result['bert_precision'] = P
        result['bert_recall'] = R
        result['bert_f1'] = F1
    if 'bleurt' in metrics and lng == 'en':
        s = bleurt(references, hypothesis, num_refs)
        result['bleurt'] = s
    if 'parent' in metrics:
        p = parent_score(references, hypothesis, graphs)
        result['parent'] = p
    if 'eredat' in metrics:
        e = eredat(hypothesis, graphs)
        result['eredat'] = e
    if 'factspotter' in metrics:
        fs = factspotter(hypothesis, graphs)
        result['factspotter'] = fs
    if 'sescore' in metrics:
        ss = sescore2(references, hypothesis, num_refs, batch_size=16)
        result['sescore2'] = ss
    if 'dqe' in metrics:
        dqe = DQE(hypothesis, graphs)
        result['dqe'] = dqe
    logging.info('FINISHING EVALUATION...')
    
    return result

def print_results(result, metrics, lng='en'):
    metrics = metrics.lower().split(',')
    headers, values = [], []
    if 'bleu' in metrics:
        # get average
        avg_bleu = sum(result['bleu']) / len(result["bleu"])

        headers.append("BLEU")
        values.append(round(avg_bleu, 3))
    if 'meteor' in metrics:
        headers.append('METEOR')
        values.append(round(result['meteor'], 3))
    if 'chrf++' in metrics:
        headers.append('chrF++')
        values.append(round(result['chrf++'], 3))
    if 'ter' in metrics:
        # get average
        avg_ter = sum(result['ter']) / len(result['ter'])

        headers.append('TER')
        values.append(round(avg_ter, 3))
    if 'bert' in metrics:
        # get average
        F1 = float(sum(result['bert_f1']) / len(result['bert_f1']))
        P = float(sum(result['bert_precision']) / len(result['bert_precision']))
        R = float(sum(result['bert_recall']) / len(result['bert_recall']))

        headers.append('BERT-SCORE P')
        values.append(round(P, 4))
        headers.append('BERT-SCORE R')  
        values.append(round(R, 4))
        headers.append('BERT-SCORE F1')
        values.append(round(F1, 4))
    if 'bleurt' in metrics and lng == 'en':
        headers.append('BLEURT')
        values.append(round(result['bleurt'], 3))
    if 'parent' in metrics:
        headers.append('PARENT')
        values.append(round(result['parent'], 3))
    if 'eredat' in metrics:
        headers.append('EREDAT')
        values.append(round(result['eredat'], 3))
    if 'factspotter' in metrics:
        headers.append('FACTSPOTTER')
        values.append(round(result['factspotter'], 3))
    if 'sescore' in metrics:
        headers.append('SESCORE2')
        values.append(round(result['sescore2'], 3))
    if 'dqe' in metrics:
        headers.append('DATA QUEST-EVAL')
        values.append(round(result['dqe'], 3))

    logging.info('PRINTING RESULTS...')
    print(tabulate([values], headers=headers))


def main():
    FORMAT = '%(levelname)s: %(asctime)-15s - %(message)s'
    logging.basicConfig(filename='eval.log', level=logging.INFO, format=FORMAT)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-ref", "--reference", help="reference translation", required=True)
    argParser.add_argument("-hyp", "--hypothesis", help="hypothesis translation", required=True)
    argParser.add_argument("-gra", "--graph", help="source graph", default=None)
    argParser.add_argument("-lng", "--language", help="evaluated language", default='en')
    argParser.add_argument("-nr", "--num_refs", help="number of references", type=int, default=4)
    argParser.add_argument("-m", "--metrics", help="evaluation metrics to be computed", default='bleu,meteor,ter,chrf++,bert,bleurt')
    argParser.add_argument("-nc", "--ncorder", help="chrF metric: character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nw", "--nworder", help="chrF metric: word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-b", "--beta", help="chrF metric: beta parameter (default=2)", type=float, default=2.0)

    args = argParser.parse_args()

    logging.info('READING INPUTS...')
    refs_path = args.reference
    hyps_path = args.hypothesis
    graph_path = args.graph
    lng = args.language
    num_refs = args.num_refs
    metrics = args.metrics

    nworder = args.nworder
    ncorder = args.ncorder
    beta = args.beta
    logging.info('FINISHING TO READ INPUTS...')

    result = run(
        refs_path=refs_path, 
        hyps_path=hyps_path, 
        graph_path=graph_path,
        num_refs=num_refs, 
        lng=lng, 
        metrics=metrics, 
        ncorder=ncorder, 
        nworder=nworder, 
        beta=beta
    )
    
    print_results(result, metrics, lng=lng)

if __name__ == '__main__':
    main()
