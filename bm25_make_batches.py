import json
from tqdm import tqdm
import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, help='the path where to save the retrieval results')
parser.add_argument('--batch_file', type=str, help='the path of the output batch file')

parser.add_argument('--num_shards', type=int, default=1)
args = parser.parse_args()

all_res = []
for i in range(args.num_shards):
    with open(os.path.join(args.results_path, 'shard%d.json'%i), 'r') as f:
        data = json.load(f)
    all_res += data

import random
random.seed(1)

S = set(range(len(all_res)))
ids = list(range(len(all_res)))
random.shuffle(ids)
p = 1

x = ids[0]
S.remove(x)
indices = [x]
for i in range(len(all_res) - 1):
    found = False
    for y in all_res[x]['retrieval'][1:]:
        if y in S:
            found = True
            x = y
            break
    if not found:
        while ids[p] not in S:
            p += 1
        x = ids[p]
    S.remove(x)
    indices.append(x)

print('total indices', len(indices))

with open(args.batch_file, 'w') as f:
    json.dump(indices, f)
