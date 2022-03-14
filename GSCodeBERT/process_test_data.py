# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import gzip
import os
import json
import numpy as np
from more_itertools import chunked
import re


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string
def tokening(string):
    new_line_list = re.findall(r"[\w']+|[^\>\<\,\\\{\}\(\)\[\]\w\s\;\:\?\"']+|[\>\<\?\,\{\}\(\)\\\[\]\;\:\"]",
                         string.strip())
    return " ".join(new_line_list)

def preprocess_test_data(test_batch_size=1000):
    print('begin____')
    DATA_DIR = './test/'
    path = 'test.json'
    # with gzip.open(path, 'r') as pf:
    #     data = pf.readlines()
    # print("len data, ", len(data))
    with open(path, 'r') as f:
        lines = f.readlines()
    f.close()

    idxs = np.arange(len(lines))
    data = np.array(lines, dtype=np.object)

    np.random.seed(0)   # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break # the last batch is smaller than the others, exclude.
        examples = []
        for d_idx, d in enumerate(batch_data): 
            # line_a = json.loads(str(d, encoding='utf-8'))
            line_a = json.loads(d)

            doc_token = tokening(format_str(line_a['docstring']))
            for dd_idx, dd in enumerate(batch_data):
                line_b = json.loads(dd)
                code_token = tokening(format_str(line_b['code']))

                example = (str(1), line_a['url'], line_b['url'], doc_token, code_token)
                example = '<CODESPLIT>'.join(example)
                examples.append(example)

        data_path = os.path.join(DATA_DIR, 'test/{}'.format('java'))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, 'batch_{}.txt'.format(batch_idx))
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))

if __name__ == '__main__':
    preprocess_test_data()
