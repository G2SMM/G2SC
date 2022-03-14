# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import os
import numpy as np
from more_itertools import chunked
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', type=int, default=1000)
    args = parser.parse_args()
    languages = ['']
    MRR_dict = {}
    top1 = 0
    top5 = 0
    top10 = 0
    cnt = 0
    for language in languages:
        file_dir = './results/{}'.format(language)
        ranks = []
        num_batch = 0
        for file in sorted(os.listdir(file_dir)):
            print(os.path.join(file_dir, file))
            with open(os.path.join(file_dir, file), encoding='utf-8') as f:
                batched_data = chunked(f.readlines(), args.test_batch_size)
                for batch_idx, batch_data in enumerate(batched_data):
                    num_batch += 1
                    cnt += 1
                    correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                    scores = np.array([float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])
                    scores2 = scores
                    rank = np.sum(scores >= correct_score)
                    ranks.append(rank)
                    sort_scores = np.sort(scores2)[::-1]
                    if np.abs(sort_scores[0] - correct_score) <= 1.0e-9:
                        top1+=1
                    if sort_scores[4] <= correct_score:
                        top5+=1
                    if sort_scores[9] <= correct_score:
                        top10+=1
        mean_mrr = np.mean(1.0 / np.array(ranks))
        print("{} mrr: {}".format(language, mean_mrr))
        MRR_dict[language] = mean_mrr

    for key, val in MRR_dict.items():
        print("{} mrr: {}".format(key, val))

    #print("cnt : ", cnt)
    print("top1 : ", top1 / cnt)
    print("top5 ", top5 / cnt)
    print("top10 ", top10 / cnt)
if __name__ == "__main__":
    main()
