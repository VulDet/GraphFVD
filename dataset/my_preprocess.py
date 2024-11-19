# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import csv
import json
import random
import pandas as pd
import sklearn

project = "NVD"
js_all = pd.read_json(open(f'{project}_PrVCs.json'))
js_all = js_all.to_dict('records')

total_num = len(js_all)
train_num = int(total_num * 0.8)
valid_num = int(total_num * 0.9)
total_idx = [i for i in range(total_num)]
random.shuffle(total_idx)

train_index = total_idx[:train_num]
valid_index = total_idx[train_num:valid_num]
test_index = total_idx[valid_num:]


with open(f'./{project}/my_train.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in train_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open(f'./{project}/my_valid.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in valid_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open(f'./{project}/my_test.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in test_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
