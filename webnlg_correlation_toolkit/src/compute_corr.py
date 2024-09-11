import json
import argparse
from tabulate import tabulate
from scipy.stats import pearsonr, spearmanr, kendalltau

from webnlg_toolkit.eval.eval import run as compute_metrics_scores

def load_processed_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def get_team_names(data):
    team_names = sorted([team_name[:-5]  for team_name in sorted(data[0].keys()) if team_name.endswith("-pred")])
    return team_names

def extract_tuple(data, year):
    tuple_dict = []
    for i, val in enumerate(data):
        if int(year) == 2020:
            for team_name in get_team_names(data):
                tuple_dict.append({
                    "team_name": team_name,
                    "triples": val['table'],
                    "refs": val['references'],
                    "pred": val[team_name + "-pred"],
                    "datacoverage": float(val[team_name + "-datacoverage"]),
                    "correctness": float(val[team_name + "-correctness"]),
                    "relevance": float(val[team_name + "-relevance"])
                    }
                )
        elif int(year) == 2017:
            for team_name in get_team_names(data):
                tuple_dict.append({
                    "team_name": team_name,
                    "triples": val['table'],
                    "refs": val['references'],
                    "pred": val[team_name + "-pred"],
                    "semantics": float(val[team_name + "-semantics"])
                    }
                )
        else:
            pass

    return tuple_dict

def eval_with_metrics(data, metrics, year):
    # Extract tuple from json data
    tuple_dict = extract_tuple(data, year)

    # Team name
    team_names = get_team_names(data)

    dict_to_run = {}
    for _, val in enumerate(tuple_dict):
        if val['team_name'] not in dict_to_run:
            if int(year) == 2020:
                dict_to_run[val['team_name']]={
                    'references': [val['refs']],
                    'predictions': [val['pred']],
                    'graphs': [val['triples']],
                    'datacoverage': [val['datacoverage']],
                    'relevance': [val['relevance']],
                    'correctness': [val['correctness']],
                }
            elif int(year) == 2017:
                dict_to_run[val['team_name']]={
                    'references': [val['refs']],
                    'predictions': [val['pred']],
                    'graphs': [val['triples']],
                    'semantics': [val['semantics']],
                }
            else:
                pass
        else:
            dict_to_run[val['team_name']]['references'].append(val['refs'])
            dict_to_run[val['team_name']]['predictions'].append(val['pred'])
            dict_to_run[val['team_name']]['graphs'].append(val['triples'])
            if int(year) == 2020:
                dict_to_run[val['team_name']]['datacoverage'].append(val['datacoverage'])
                dict_to_run[val['team_name']]['correctness'].append(val['correctness'])
                dict_to_run[val['team_name']]['relevance'].append(val['relevance'])
            elif int(year) == 2017:
                dict_to_run[val['team_name']]['semantics'].append(val['semantics'])
            else:
                pass
            
    # Compute metric scores
    metric_results = {}
    for team in team_names:
        metric_results[team] = compute_metrics_scores(
            refs_path=dict_to_run[team]['references'],
            hyps_path=dict_to_run[team]['predictions'],
            graph_path=dict_to_run[team]['graphs'],
            num_refs=4,
            lng='en',
            metrics=metrics,
            ncorder=6,
            nworder=2,
            beta=2,
        )

        # Add human annotation scores
        if int(year) == 2020:
            metric_results[team]['datacoverage'] = dict_to_run[team]['datacoverage']
            metric_results[team]['correctness'] = dict_to_run[team]['correctness']
            metric_results[team]['relevance'] = dict_to_run[team]['relevance']
        if int(year) == 2017:
            metric_results[team]['semantics'] = dict_to_run[team]['semantics']
    return metric_results

def print_results(metric_results, metrics, human_metrics):
    team_names = metric_results.keys()
    headers_sys, headers_sent, system_scores, sent_scores = [], [], [], []

    headers_sys.extend(["System Level Correlation"])
    headers_sent.extend(["Sentence Level Correlation"])

    metrics = metrics.lower().split(',')
    if 'bleu' in metrics:
        results = {}
    
        for hm in human_metrics:
            sent_bleus = []
            sent_human_scores = []
            sys_bleus = []
            sys_human_scores = []
            for team in team_names:
                # Compute system level correlation
                avg_team_bleu = sum(metric_results[team]['bleu']) / len(metric_results[team]['bleu'])
                avg_hm_score = sum(metric_results[team][hm]) / len(metric_results[team][hm])
                sys_bleus.append(avg_team_bleu)
                sys_human_scores.append(avg_hm_score)
                # Compute sentence level correlation
                sent_bleus.extend(metric_results[team]['bleu'])
                sent_human_scores.extend(metric_results[team][hm])
            
            results[hm, 'sent_level'] = pearsonr(sent_bleus, sent_human_scores)[0]
            results[hm, 'system_level'] = pearsonr(sys_bleus, sys_human_scores)[0]

        # Draw tabualte
        headers_sys.extend(["BLEU"])
        headers_sent.extend(["BLEU"])
        for hm in human_metrics:
            system_scores.append((hm, round(results[hm, 'system_level'], 3)))
            sent_scores.append((hm, round(results[hm, 'sent_level'], 3)))

        # print(tabulate(system_scores, headers=headers_sys, tablefmt='grid'))
        # print(tabulate(sent_scores, headers=headers_sent, tablefmt='grid'))

    if 'eredat' in metrics:
        results = {}
    
        for hm in human_metrics:
            sent_eredat = []
            sent_human_scores = []
            sys_eredat = []
            sys_human_scores = []
            for team in team_names:
                # Compute system level correlation
                avg_team_score = metric_results[team]['eredat'].mean()
                avg_hm_score = sum(metric_results[team][hm]) / len(metric_results[team][hm])
                sys_eredat.append(avg_team_score)
                sys_human_scores.append(avg_hm_score)
                # Compute sentence level correlation
                sent_eredat.extend(metric_results[team]['eredat'].tolist())
                sent_human_scores.extend(metric_results[team][hm])
            
            results[hm, 'sent_level'] = pearsonr(sent_eredat, sent_human_scores)[0]
            results[hm, 'system_level'] = pearsonr(sys_eredat, sys_human_scores)[0]

        # Draw tabualte
        headers_sys.extend(["EREDAT"])
        headers_sent.extend(["EREDAT"])
        for hm in human_metrics:
            # Check if this human metric already in the table
            exist = False
            for i, pair in enumerate(system_scores):
                if pair[0] == hm:
                    exist = True
                    # Add new element
                    system_scores[i] = pair + (round(results[hm, 'system_level'], 3),)
                    sent_scores[i] = sent_scores[i] + (round(results[hm, 'sent_level'], 3),)
            
            if not exist:
                system_scores.append((hm, round(results[hm, 'system_level'], 3)))
                sent_scores.append((hm, round(results[hm, 'sent_level'], 3)))

    if 'factspotter' in metrics:
        results = {}
    
        for hm in human_metrics:
            sent_fs = []
            sent_human_scores = []
            sys_fs = []
            sys_human_scores = []
            for team in team_names:
                # Compute system level correlation
                avg_team_score = sum(metric_results[team]['factspotter'])/len(metric_results[team]['factspotter'])
                avg_hm_score = sum(metric_results[team][hm]) / len(metric_results[team][hm])
                sys_fs.append(avg_team_score)
                sys_human_scores.append(avg_hm_score)
                # Compute sentence level correlation
                sent_fs.extend(metric_results[team]['factspotter'])
                sent_human_scores.extend(metric_results[team][hm])
            
            results[hm, 'sent_level'] = pearsonr(sent_fs, sent_human_scores)[0]
            results[hm, 'system_level'] = pearsonr(sys_fs, sys_human_scores)[0]

        # Draw tabualte
        headers_sys.extend(["FACTSPOTTER"])
        headers_sent.extend(["FACTSPOTTER"])
        for hm in human_metrics:
            # Check if this human metric already in the table
            exist = False
            for i, pair in enumerate(system_scores):
                if pair[0] == hm:
                    exist = True
                    # Add new element
                    system_scores[i] = pair + (round(results[hm, 'system_level'], 3),)
                    sent_scores[i] = sent_scores[i] + (round(results[hm, 'sent_level'], 3),)
            
            if not exist:
                system_scores.append((hm, round(results[hm, 'system_level'], 3)))
                sent_scores.append((hm, round(results[hm, 'sent_level'], 3)))

    if 'dqe' in metrics:
        results = {}
    
        for hm in human_metrics:
            sent_dqe = []
            sent_human_scores = []
            sys_dqe = []
            sys_human_scores = []
            for team in team_names:
                # Compute system level correlation
                avg_team_score = metric_results[team]['dqe']['corpus_score'].mean()
                avg_hm_score = sum(metric_results[team][hm]) / len(metric_results[team][hm])
                sys_dqe.append(avg_team_score)
                sys_human_scores.append(avg_hm_score)
                # Compute sentence level correlation
                sent_dqe.extend(metric_results[team]['dqe']['ex_level_scores'])
                sent_human_scores.extend(metric_results[team][hm])
            
            results[hm, 'sent_level'] = pearsonr(sent_dqe, sent_human_scores)[0]
            results[hm, 'system_level'] = pearsonr(sys_dqe, sys_human_scores)[0]

        # Draw tabualte
        headers_sys.extend(["DATA QUEST-EVAL"])
        headers_sent.extend(["DATA QUEST-EVAL"])
        for hm in human_metrics:
            # Check if this human metric already in the table
            exist = False
            for i, pair in enumerate(system_scores):
                if pair[0] == hm:
                    exist = True
                    # Add new element
                    system_scores[i] = pair + (round(results[hm, 'system_level'], 3),)
                    sent_scores[i] = sent_scores[i] + (round(results[hm, 'sent_level'], 3),)
            
            if not exist:
                system_scores.append((hm, round(results[hm, 'system_level'], 3)))
                sent_scores.append((hm, round(results[hm, 'sent_level'], 3)))

    # Table visualization
    print(tabulate(system_scores, headers=headers_sys, tablefmt='grid'))
    print(tabulate(sent_scores, headers=headers_sent, tablefmt='grid'))


def main():
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-i", 
        "--input_path", 
        help="processed file path", 
        default="webnlg_correlation_toolkit/data/human_eval_webnlg_20.json"
    )
    argParser.add_argument("-m", "--metrics", help="evaluation metrics to be computed", default='bleu,meteor,ter,chrf++,bert,bleurt')
    argParser.add_argument("-y", "--year", help="specify webnlg version (2017 or 2020)", default='2020')

    args = argParser.parse_args()
    input_path = args.input_path
    metrics = args.metrics
    year = args.year
    if int(year) == 2020:
        human_metrics = ['datacoverage', 'relevance', 'correctness']
    elif int(year) == 2017:
        human_metrics = ['semantics']
    else:
        pass

    # Load json data
    data = load_processed_json(input_path)

    # Eval data with the metrics
    metric_results = eval_with_metrics(data, metrics, year)

    # Print results
    print_results(metric_results, metrics, human_metrics)


if __name__ == '__main__':
    main()