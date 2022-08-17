from functools import reduce
import json
import os
import re
import tarfile
import tempfile

from itertools import chain
from collections import defaultdict

import numpy as np
np.random.seed(1337)

#################################################################
#                       Loader MSR corpus                      #
#################################################################

def loader_msr(path):
    data = {}
    
    train_groups = detected_groups(get_data_msr(path + 'msr_paraphrase_train.txt'))
    test_groups = detected_groups(get_data_msr(path + 'msr_paraphrase_test.txt'))
    train_groups = list(filter(lambda x: len(x) >= 2 and len(x) < 100, train_groups))
    test_groups = list(filter(lambda x: len(x) >= 2 and len(x) < 100, test_groups))

    data['train-gr'] = train_groups
    data['test-gr'] = test_groups

    data['train-y'] = np.repeat(range(len(train_groups)), list(map(len, train_groups)))
    data['test-y'] = np.repeat(range(len(test_groups)), list(map(len, test_groups)))    

    data['train-x'] = list(chain.from_iterable(train_groups))
    data['test-x'] = list(chain.from_iterable(test_groups))

    return data

def yield_examples_msr(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if i == 0:
            continue
        if limit and i > limit:
            break
        data = line.split('\t')
        label = data[0]
        s1 = data[3]
        s2 = data[4][:-1]
        yield (label, s1, s2)

def get_data_msr(fn, limit=None):
    raw_data = list(yield_examples_msr(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    # print(max(len(x.split()) for x in left))
    # print(max(len(x.split()) for x in right))

    LABELS = {'0': 0, '1': 1}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
#     Y = np_utils.to_categorical(Y, len(LABELS))
    
#     print(len(left), len(right), len(Y))
    
    return left, right, Y


#################################################################
#                       Loader SNLI corpus                      #
#################################################################

def loader_snli(path):
    data = {}

    train_groups = detected_groups(get_data(path + 'snli_1.0_train.jsonl'))
    test_groups = detected_groups(get_data(path + 'snli_1.0_test.jsonl'))
    train_groups = list(filter(lambda x: len(x) >= 7 and len(x) < 50, train_groups))
    test_groups = list(filter(lambda x: len(x) >= 3 and len(x) < 50, test_groups))

    data['train-gr'] = train_groups
    data['test-gr'] = test_groups

    data['train-y'] = np.repeat(range(len(train_groups)), list(map(len, train_groups)))
    data['test-y'] = np.repeat(range(len(test_groups)), list(map(len, test_groups)))    

    data['train-x'] = list(chain.from_iterable(train_groups))
    data['test-x'] = list(chain.from_iterable(test_groups))

    return data

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['gold_label']
        s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
        if skip_no_majority and label == '-':
            continue
        yield (label, s1, s2)

def get_data(fn, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    # print(max(len(x.split()) for x in left))
    # print(max(len(x.split()) for x in right))

    LABELS = {'contradiction': 0, 'neutral': 0, 'entailment': 1}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
#     Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y




#################################################################
#                          Tools                                #
#################################################################

def detected_groups(data):
    graph_dict = defaultdict(set)

    for left, right, label in zip(*data):
        if label == 1:
            graph_dict[left].add(right)
            graph_dict[right].add(left)

    visited_nodes = set()
    
    def DFS(start, comp):
        comp.add(start)
        visited_nodes.add(start)
        sim = graph_dict[start]
        if not all(map(lambda x: x in visited_nodes, sim)):
            for item in sim:
                if item not in visited_nodes:
                    comp.update(DFS(item, comp))
        return comp
    
    components = []

    for item in data[0]:
        if item not in visited_nodes:
            components.append(DFS(item, set()))


    for item in data[1]:
        if item not in visited_nodes:
            components.append(DFS(item, set()))
            
    return components