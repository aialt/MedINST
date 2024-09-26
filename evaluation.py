import argparse
import json
import os
import random
import time

from tqdm import tqdm
import datasets
from openai import OpenAI
import numpy as np
from typing import List, Dict
import re
import string
from sklearn.metrics import f1_score
from collections import Counter

tests = {'QA': ['BioASQ-Task-B-yesno', 'PubMedQA-labeled', 'MedQA'],
         'TE': ['SciFact', 'ManConCorpus', 'CoVERt'],
         'NER': ['NCBI-disease',
                 'BC5CDR',
                 'AnEM',
                 'BioNLP-2009',
                 'BioNLP-2011-GE',
                 'BioNLP-2011-ID',
                 'BioNLP-2011-REL',
                 'BioNLP-2013-CG',
                 'BioNLP-2013-GE',
                 'BioNLP-2013-GRO',
                 'BioNLP-2013-PC',
                 'BioRED',
                 'tmVar-v3'],
         'TXTCLASS': ['Hallmarks-of-Cancer', 'MedDialog'],
         'NED': ['MeDAL', 'tmVar-v3-NED'],
         'RE': ['AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE'],
         'COREF': ['AnEM-COREF', 'MLEE-COREF'],
         'SUM': ['Multi-XScience'],
         'EE': ['MLEE-EE'],
         'STS': ['BIOSSES'],
         'TRANSL': ['ParaMed']}

bio = ['NCBI-disease', 'BC5CDR']
cls = ['BioASQ-Task-B-yesno', 'PubMedQA-labeled', 'SciFact', 'ManConCorpus', 'CoVERt', 'MedDialog']
entity = ['AnEM-COREF', 'MLEE-COREF', 'tmVar-v3-NED', 'AnEM', 'BioNLP-2009', 'BioNLP-2011-GE', 'BioNLP-2011-ID',
          'BioNLP-2011-REL', 'BioNLP-2013-CG',
          'BioNLP-2013-GE', 'BioNLP-2013-GRO', 'BioNLP-2013-PC', 'BioRED',
          'tmVar-v3', 'AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE',
          'MLEE-EE'
          ]
em = ['MedQA', 'MeDAL']
mse = ['BIOSSES']
multicls = ['Hallmarks-of-Cancer']


def mse_score(targets, preds):
    def extract_integers_from_string(s):
        integers = re.findall(r'\d+', s)
        integers = [int(num) for num in integers]
        return list(set(integers))

    ts = []
    ps = []
    for t, p in zip(targets, preds):
        t_numbers = extract_integers_from_string(t)
        p_numbers = extract_integers_from_string(p)
        if len(t_numbers) != 1 or len(p_numbers) != 1:
            t_num = 0
            p_num = 5
        elif t_numbers[0] not in [0, 1, 2, 3, 4, 5] or p_numbers[0] not in [0, 1, 2, 3, 4, 5]:
            t_num = 0
            p_num = 5
        else:
            t_num = t_numbers[0]
            p_num = p_numbers[0]
        ts.append(t_num)
        ps.append(p_num)
    n = len(ts)
    # print(ts)
    # print(ps)
    mse = sum((x - y) ** 2 for x, y in zip(ts, ps)) / n
    return mse


def post_bio(target, pred):
    def extract_tags(input_string):
        pattern = r'\[B\]|\[I\]|\[O\]'
        matches = re.findall(pattern, input_string)
        return matches

    perd_labels = extract_tags(pred)
    target_labels = extract_tags(target)
    return target_labels, perd_labels


def post_entity(target, pred):
    def extract_entities_with_stack(s):
        stack = []
        entities = []
        current_entity = []
        for char in s:
            if char == '[':
                if stack:
                    current_entity.append(char)
                stack.append(char)
            elif char == ']':
                if not stack:
                    current_entity = []
                    continue
                stack.pop()
                if stack:
                    current_entity.append(char)
                else:
                    entities.append(normalize_answer(''.join(current_entity)))
                    current_entity = []
            elif stack:
                current_entity.append(char)
        return entities

    target_entities = extract_entities_with_stack(target)
    pred_entities = extract_entities_with_stack(pred)
    return target_entities, pred_entities


def label_level_f1(targets, preds):
    macro_f1 = f1_score(targets, preds, labels=sorted(set(targets)), average='macro')
    return macro_f1


def entity_level_f1(target, pred):
    
    true_counter = Counter(target)
    pred_counter = Counter(pred)

   
    tp = sum((true_counter & pred_counter).values())

   
    fp = sum((pred_counter - true_counter).values())

    
    fn = sum((true_counter - pred_counter).values())


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return score


def rouge(prediction: str, ground_truth: str, rouge_metric):
    score = rouge_metric.compute(
        predictions=[prediction],
        references=[ground_truth],
        **{'use_aggregator': False, 'use_stemmer': True, 'rouge_types': ['rougeL']}
    )
    return score['rougeL'][0].fmeasure


def word_level_f1(prediction: str, ground_truth: str):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(targets: List[str], predictions: List[str], evaluation_types: List[str], rouge_metric) -> Dict:
    assert len(predictions) == len(targets), \
        f"The pred file does not have the same length as the gold data: {len(targets)} vs {len(predictions)}"

    metrics = {}

    for idx, (gold, pred) in datasets.tqdm(enumerate(zip(targets, predictions))):

        if 'rouge' not in metrics:
            metrics['rouge'] = 0
        metrics['rouge'] += rouge(pred, gold, rouge_metric)

        if 'f1' not in metrics:
            metrics['f1'] = 0
        metrics['f1'] += word_level_f1(pred, gold)

        if 'entity' in evaluation_types:
            ts, ps = post_entity(gold, pred)
            if 'entity_level_f1' not in metrics:
                metrics['entity_level_f1'] = 0
            metrics['entity_level_f1'] += entity_level_f1(ts, ps)

        if 'multicls' in evaluation_types:
            ts = gold.split(', ')
            ps = pred.split(', ')
            ts = [t.lower().strip() for t in ts]
            ps = [p.lower().strip() for p in ps]
            if 'entity_level_f1' not in metrics:
                metrics['entity_level_f1'] = 0
            metrics['entity_level_f1'] += entity_level_f1(ts, ps)

        if 'em' in evaluation_types:
            if 'em' not in metrics:
                metrics['em'] = 0
            metrics['em'] += exact_match_score(pred, gold)

    # e.g., selecting A, B, C, etc.
    # normalize tne metrics
    for key in metrics.keys():
        metrics[key] /= len(predictions)

    if 'bio' in evaluation_types:
        ts = []
        ps = []
        for t, p in zip(targets, predictions):
            post_t, post_p = post_bio(t, p)
            # if len(post_t) != len(post_p):
            #     print('#################')
            #     print(t)
            #     print(post_t)
            #     print('----------------')
            #     print(p)
            #     print(post_p)
            #     print('#################')
            #     exit()
            if len(post_t) > len(post_p):
                post_p = post_p + ['N' for _ in range(len(post_t) - len(post_p))]
            else:
                post_p = post_p[:len(post_t)]
            ts.extend(post_t)
            ps.extend(post_p)
        metrics['label_leval_f1'] = label_level_f1(ts, ps)

    if "cls" in evaluation_types:
        ts = [normalize_answer(t) for t in targets]
        ps = [normalize_answer(p) for p in predictions]
        metrics['label_level_f1'] = label_level_f1(ts, ps)

    if 'mse' in evaluation_types:
        metrics['mse'] = mse_score(targets, predictions)

    return metrics


def predict(ex, client, model, flag=False):
    messages = []
    if 'history' in ex:
        for h in ex['history']:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
    prompt = ex['instruction'] + '\n' + ex['input']
    messages.append({"role": "user", "content": prompt})

    # prompt = ex['instruction']
    # if 'history' in ex:
    #     for i, h in enumerate(ex['history']):
    #         prompt += f'\nExample {i + 1}:\nInput: {h[0].replace(ex["instruction"],"")}\nOutput: {h[1]}'
    # prompt += f'\nNow complete the new example:\nInput: {ex["input"]}\nOutput:'
    # messages.append({'role': 'user', 'content': prompt})
    try_times = 10
    while try_times > 0:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0
            )
            response, num_input, num_output = completion.choices[
                0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens
            if flag:
                print(messages)
                print(completion.choices[0].message.content)
            return response, num_input, num_output
        except:
            print("Waiting for the server...")
            time.sleep(30)
            try_times -= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument('--zero', action='store_true')
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()
    name = args.name
    dir = args.dir
    key = args.key
    model = args.model
    base_url = args.base_url
    zero = args.zero
    client = OpenAI(api_key=key, base_url=base_url)
    rouge_metric = datasets.load_metric('rouge', experiment_id=str(random.randint(1, 888888)))
    with open(f'{dir}/{name}.json', 'w', encoding='utf-8') as out:
        results = {}
        for cat, test_names in tests.items():
            results[cat] = {}
            for test in test_names:
                flag = True
                print(f'------------------------ {cat}: {test} ------------------------')
                results[cat][test] = {}
                config = '-zs' if zero else ''
                data = datasets.load_dataset('LiinXemmon/MedINST32', test+config)['test']
                targets = []
                predictions = []
                tqdm_data = tqdm(data)
                for d in tqdm_data:
                    targets.append(d['output'])
                    if not flag:
                        flag = random.randint(1, 50) == 1
                    pre, len_prompt, len_gen = predict(d, client, model, flag=flag)
                    predictions.append(pre)
                    tqdm_data.set_description(f'Inp: {len_prompt} Gen: {len_gen}')
                    flag = False
                # print(targets, predictions)
                results[cat][test]['generated'] = [{'prediction': pre, 'target': target} for pre, target in
                                                   zip(predictions, targets)]
                types = []
                if test in cls:
                    types.append('cls')
                elif test in em:
                    types.append('em')
                elif test in entity:
                    types.append('entity')
                elif test in multicls:
                    types.append('multicls')
                elif test in bio:
                    types.append('bio')
                elif test in mse:
                    types.append('mse')

                results[cat][test]['metrics'] = evaluate(targets, predictions, types, rouge_metric)
                print(f"{test}: ", results[cat][test]['metrics'])
        json.dump(results, out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
