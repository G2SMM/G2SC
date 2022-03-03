#coding=utf-8

import json
from transformers import AutoTokenizer
import re
import numpy as np
### txt2json
def load2():
    with open('s_code.txt', 'r') as f:
        train_codes = f.readlines()
    f.close()
    with open('s_desc.txt', 'r') as f:
        train_desc = f.readlines()
    f.close()
    jsons = []
    for i in range(len(train_codes)):
        sample = {}
        sample['url'] = str(i)
        sample['code'] = train_codes[i]
        sample['docstring'] = train_desc[i]
        sample['func_name'] = str(i)
        json_text = json.dumps(sample)
        jsons.append(json_text + "\n")
    
    with open('s_net.json', 'w+', encoding='utf-8') as f:
        f.writelines(jsons)
    f.close()


# 8:1:1 split dataset
def load1(path):

    with open(path, 'r') as f:
        lines = f.readlines()
    f.close()
    #lines = lines[11000:12000]
    train_data = []
    val_data = []
    test_data = []
    lens = len(lines)
    for i in range(lens):
        #if i >= 0 and i <= lens * 0.8:
        line = json.loads(lines[i])
        #line['url'] = str(i)
        if i >= 0 and i < lens * 0.8:
            train_data.append(json.dumps(line) + "\n")
        elif i >= lens * 0.8 and i < lens * 0.9:
            val_data.append(json.dumps(line) + "\n")
        elif i >= lens * 0.9 and i < lens:
            test_data.append(json.dumps(line) + "\n")
    with open('s_train.json', 'w+', encoding='utf-8') as f:
        f.writelines(train_data)
    f.close()

    with open('s_val.json', 'w+', encoding='utf-8') as f:
        f.writelines(val_data)
    f.close()

    with open('s_test.json', "w+", encoding='utf-8') as f:
        f.writelines(test_data)
    f.close()
    



# build train/valid dataset and balance pos and neg
# codebert train/valid 格式：(label, url, class.method, docstring, code) <CODESPLIT>
# our：(label, index, methodname, docstring, code/dfs),
def build(data_path, name):
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
        print(ix)
        pos_index = item['url']
        pos_methodname = item['func_name']
        pos_code = tokening(format_str(item['dfs']))
        pos_docstring = tokening(format_str(item['docstring']))
        pos = (str(1), pos_index, pos_methodname, pos_docstring, pos_code)
        new_datas.append('<CODESPLIT>'.join(pos) + "\n")
        while True:
            neg_ix = np.random.randint(0, length)
            if neg_ix != ix: # random choose 
                break
        neg_index = pos_index+'_'+js[neg_ix]['url']
        neg_methodname = js[neg_ix]['func_name']
        neg_code = tokening(format_str(js[neg_ix]['dfs']))
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
    build("new_val.json", "desc_dfs_valid")
