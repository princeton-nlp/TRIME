import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary
import faiss
import random

import torch.nn.functional as F

from tqdm import tqdm

class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.index = self.setup_faiss(args)

        self.temp = args.softmax_temp

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        start = time.time()
        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'-fp16_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
        print('Reading keys took {} s'.format(time.time() - start))

        self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = self.vals_from_memmap[range(self.dstore_size)]

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                if args.dstore_fp16:
                    self.keys_from_memmap = np.memmap(args.dstore_filename+'-fp16_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                    self.keys = self.keys_from_memmap[range(self.dstore_size)]
                else:
                    self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                    self.keys = self.keys_from_memmap[range(self.dstore_size)]
                print('Keys shape:', self.keys.shape)

            print('Loading to memory took {} s'.format(time.time() - start))

        if not args.use_faiss_cpu:
            self.cpu_index = index
            print('Index: CPU to GPU (device 1, 2, ...)')
            res = faiss.StandardGpuResources()
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            ngpus = faiss.get_num_gpus()
            print(f"ngpus = {ngpus}")
            index = faiss.index_cpu_to_gpu_multiple_py([faiss.StandardGpuResources() for _ in range(1, ngpus)], index, co, range(1, ngpus))
            print('Index: on GPU now')

        return index


    def get_knns(self, queries):
        q_vecs = queries.detach().cpu().float().numpy()
        if self.metric_type == 'cos':
            faiss.normalize_L2(q_vecs)
        dists, knns = self.index.search(q_vecs, self.k)
        return dists, knns

    def dist_func(self, d, k, q, function=None):
        # function: 'l2', 'ip', or 'cos'
        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            # Default behavior for COS metric is to return faiss distances.
            qsize = q.shape
            if self.metric_type == 'l2':
                knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                return -1 * l2
            return d

        if function == 'ip' or function == 'cos':
            qsize = q.shape
            knns_vecs = torch.from_numpy(self.keys[k]).cuda()
            if self.half:
                knns_vecs = knns_vecs.half()

            q_vecs = q.view(qsize[0], 1, qsize[1])

            if function == 'cos':
                q_vecs = F.normalize(q_vecs, p=2, dim=-1)

            d = (knns_vecs * q_vecs).sum(dim=-1)

            return d

        if function == 'do_not_recomp_l2':
            return -1 * d

        raise ValueError("Invalid knn similarity function!")

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        # queries are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.contiguous().view(-1, qshape[-1])
        tgt = tgt.contiguous().view(-1)
        dists, knns = self.get_knns(queries[tgt != pad_idx])
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        dists = self.dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)
        vals = self.vals[knns]

        probs = utils.log_softmax(dists / self.temp, dim=-1)

        index_mask = torch.eq(torch.from_numpy(vals).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000.0).cuda()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)
