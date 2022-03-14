#coding=utf-8

import json
from transformers import AutoTokenizer
import re
import numpy as np

# build train/valid dataset and balance pos and neg
# codebert train/valid 格式：(label, url, class.method, docstring, code) <CODESPLIT>
# our：(label, index, methodname, docstring, code/dfs),
def build(data_path, name, type='code'):
    new_datas = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
    f.close()
    js = []
    for line in lines:
       js.append(json.loads(line)) 
    length = len(js)
    d = {}
    for ix, item in enumerate(js):
        #print(ix)
        pos_index = item['url']
        pos_methodname = item['func_name']
        pos_code = tokening(format_str(item[type]))
        pos_docstring = tokening(format_str(item['docstring']))
        pos = (str(1), pos_index, pos_methodname, pos_docstring, pos_code)
        new_datas.append('<CODESPLIT>'.join(pos) + "\n")
        while True:
            neg_ix = np.random.randint(0, length)
            if neg_ix != ix: # random choose 
                break
        neg_index = pos_index+'_'+js[neg_ix]['url']
        neg_methodname = js[neg_ix]['func_name']
        neg_code = tokening(format_str(js[neg_ix][type]))
        #neg_docstring = tokening(format_str(js[neg_ix]['docstring']))
        neg = (str(0), neg_index, neg_methodname, pos_docstring, neg_code)
        new_datas.append('<CODESPLIT>'.join(neg) + "\n")
    np.random.seed(0)
    idxs = np.arange(len(new_datas))
    new_datas = np.array(new_datas, dtype=np.object)
    np.random.shuffle(idxs)
    new_datas = new_datas[idxs]
    with open(name + '.txt', 'w+') as f:
        f.writelines(new_datas)
    f.close()

# 
def tokening(string):
    new_line_list = re.findall(r"[\w']+|[^\>\<\,\\\{\}\(\)\[\]\w\s\;\:\?\"']+|[\>\<\?\,\{\}\(\)\\\[\]\;\:\"]",
                         string.strip())
    return " ".join(new_line_list)

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string



if __name__ == "__main__":
    build("train.json", "desc_code_train","code")
    build("valid.json", "desc_code_valid", "code")
    build("train_gs.json", "desc_gs_train", "dfs")
    build("valid_gs.json", "desc_gs_valid", "dfs")
