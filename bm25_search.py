from pyserini.search.lucene import LuceneSearcher
import json
import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index_path', type=str, help='the path of BM25 index')
parser.add_argument('--segments_path', type=str, help='the path of the file storing all the segments')
parser.add_argument('--results_path', type=str, help='the path where to save the retrieval results')

parser.add_argument('--num_shards', type=int, default=1)
parser.add_argument('--shard_id', type=int, default=0)

args = parser.parse_args()

print('num shards: {}; shard id: {}'.format(args.num_shards, args.shard_id))
with open(args.segments_path, 'r') as f:
    segments = json.load(f)

searcher = LuceneSearcher(args.index_path)

N = len(segments)
N_per_S = (N + args.num_shards - 1) // args.num_shards

ret_results = []
for b in segments[N_per_S * args.shard_id : N_per_S * (args.shard_id + 1)]:
    try:
        hits = searcher.search(b['contents'], 20)
        ret = {'id': b['id'], 'retrieval': [h.docid for h in hits]}
    except e:
        print('error!', e)
        ret = {'id': b['id'], 'retrieval': [b['id']]}
    ret_results.append(ret)

if not os.path.exists(args.results_path):
    os.mkdir(args.results_path)

with open(os.path.join(args.results_path, 'shard%d.json'%args.shard_id), 'w') as f:
    json.dump(ret_results, f)
