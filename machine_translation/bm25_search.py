from pyserini.search.lucene import LuceneSearcher
import json
from tqdm import tqdm
import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index_path', type=str, help='the path of BM25 index')
parser.add_argument('--samples_path', type=str, help='the path of the file storing all the segments')
parser.add_argument('--results_path', type=str, help='the path where to save the retrieval results')

parser.add_argument('--num_shards', type=int, default=1)
parser.add_argument('--shard_id', type=int, default=0)

args = parser.parse_args()

print('num shards: {}; shard id: {}'.format(args.num_shards, args.shard_id))
with open(args.samples_path, 'r') as f:
    blocks = json.load(f)

with open(args.samples_path, 'r') as f:
    blocks = json.load(f)

id2c = {int(x['id']): x['contents'] for x in blocks}

searcher = LuceneSearcher(args.index_path)

N = len(blocks)
N_per_S = (N + args.num_shards - 1) // args.num_shards

ret_results = []
for b in tqdm(blocks[N_per_S * args.shard_id: N_per_S * (args.shard_id + 1)]):
    try:
        hits = searcher.search(b['contents'], 20)
        ret = {'id': b['id'], 'retrieval': [int(h.docid) for h in hits]}
    except:
        print('error!')
        ret = {'id': b['id'], 'retrieval': [int(b['id'])]}
    ret_results.append(ret)

if not os.path.exists(args.results_path):
    os.mkdir(args.results_path)

with open(os.path.join(args.results_path, 'shard%d.json'%args.shard_id), 'w') as f:
    json.dump(ret_results, f)
