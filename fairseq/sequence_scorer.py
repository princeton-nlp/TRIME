# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from threading import local
import torch
import sys
import numpy as np
import time
import math

from fairseq import utils
from fairseq.data import Dictionary

import torch.nn.functional as F


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, args=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args

        self.mem = None
        self.last_ids = None
        self.tot_mem = 0

    @torch.no_grad()
    def trime_generate(self, models, sample, dstore=None, mode=None, use_local=False, use_long=False, use_external=False, use_interp=False):
        """Score a batch of translations. (use ctx CE loss)"""
        net_input = sample['net_input']

        def retrieve_from_ds(dstore, queries, tgt, temp=None):
            dists, knns = dstore.get_knns(queries[tgt != self.pad])
            dists = torch.from_numpy(dists).cuda()
            dists = dstore.dist_func(dists, knns, queries[tgt != self.pad, :], function=dstore.sim_func)
            dists = dists / math.sqrt(queries.shape[-1]) / (self.args.softmax_temp if temp is None else temp)
            vals = dstore.vals[knns]
            vals = torch.from_numpy(vals).cuda().squeeze()

            full_dists = torch.full((queries.shape[0], dstore.k), -10000.0).cuda()
            full_dists[tgt != self.pad] = dists.float()
            full_vals = torch.full((queries.shape[0], dstore.k), -1).cuda()
            full_vals[tgt != self.pad] = vals

            return full_dists, full_vals

        def retrieve_logits(reps, labels, cache_temp=None):
            assert use_local or use_long or use_external

            reps = reps.contiguous().view(-1, reps.shape[-1])
            labels = labels.contiguous().view(-1)
            bsz = reps.shape[0]

            assert (not use_long) or use_local

            if use_local and use_long:
                if self.mem is not None:
                    inbatch_reps = torch.cat((self.mem[0], reps), dim=0)
                    inbatch_labels = torch.cat((self.mem[1], labels), dim=0)
                else:
                    inbatch_reps = reps
                    inbatch_labels = labels
            elif use_local:
                inbatch_reps = reps
                inbatch_labels = labels
            else:
                # only external memory is used -- directly return
                assert use_external
                ds_logits, ds_labels = retrieve_from_ds(dstore, reps, labels, cache_temp)
                return ds_logits, ds_labels

            local_logits = torch.mm(reps, inbatch_reps.T) / math.sqrt(reps.shape[-1]) / (self.args.softmax_temp if cache_temp is None else cache_temp)

            temp_mask = torch.ones((bsz, bsz + self.args.mem_size), device=reps.device)
            local_mask = torch.triu(temp_mask, diagonal=self.args.mem_size)
            local_mask = local_mask[:, -local_logits.shape[1]:]
            local_mask = local_mask * -10000.0

            local_logits = local_logits + local_mask
            
            local_labels = inbatch_labels.view(1, -1).expand(bsz, -1)

            if use_external:
                ds_logits, ds_labels = retrieve_from_ds(dstore, reps, labels, cache_temp)
                ret_logits = torch.cat((local_logits, ds_logits), dim=-1)
                ret_labels = torch.cat((local_labels, ds_labels), dim=-1)
            else:
                ret_logits = local_logits
                ret_labels = local_labels

            return ret_logits, ret_labels

        def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
            combine_probs = torch.stack([vocab_p.view(1, -1), knn_p.view(1, -1)], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - coeff)
            coeffs[1] = np.log(coeff)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

            return curr_prob

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None

        assert(len(models) == 1)
        for model in models:
            model.eval()
            net_output = model(**net_input)
            orig_target = model.get_targets(sample, net_output)

            if hasattr(model.decoder, 'adaptive_softmax') and model.decoder.adaptive_softmax is not None:
                logits, target, target_idxs = model.decoder.adaptive_softmax(net_output[0], orig_target.view(-1), return_target_idx=True)
                assert len(target) == len(logits)
                norm_t = torch.logsumexp(logits[0], dim=-1)
                token_loss = F.cross_entropy(logits[0], target[0], ignore_index=self.pad, reduction='none')
                for i in range(len(target_idxs)):
                    if target_idxs[i] is not None:
                        token_loss[target_idxs[i]] += F.cross_entropy(logits[i + 1], target[i + 1], ignore_index=self.pad, reduction='none')
            else:
                logits = model.decoder.get_logits(net_output, sample)
                logits = logits.reshape(-1, logits.shape[-1])
                ori_probs = F.log_softmax(logits, dim=-1)
                token_loss = -ori_probs.gather(dim=1, index=orig_target.view(-1, 1)).view(-1)
                norm_t = torch.logsumexp(logits, dim=-1)
            
            self.check_mem(sample)

            if use_local or use_long or use_external:
                reps = net_output[1][self.args.knn_keytype].permute(1, 0, 2)
                rep_labels = orig_target.clone()
                virtual_labels = sample['net_input']['src_tokens'].roll(-1, -1)
                rep_labels[orig_target == self.pad] = virtual_labels[orig_target == self.pad]

                # retrieve logits from local, long, and/or external
                aug_logits, aug_labels = retrieve_logits(reps, rep_labels)
                norm_c = torch.logsumexp(aug_logits, dim=-1)

                in_batch_probs = F.log_softmax(aug_logits, dim=-1)
                negatives = (orig_target.view(-1, 1) != aug_labels)
                ctx_prob = torch.logsumexp(in_batch_probs + negatives * -10000.0, dim=-1)

                norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)

                if mode == 'gn':
                    avg_probs = torch.logsumexp(torch.stack((-token_loss + norm_t - norm_tpc, ctx_prob + norm_c - norm_tpc), dim=-1), dim=-1)
                elif mode == 'li':
                    avg_probs = combine_knn_and_vocab_probs(ctx_prob, -token_loss, self.args.cache_lmbda)
                else:
                    raise NotImplementedError
            else:
                avg_probs = -token_loss

            if use_interp:
                probs = avg_probs
                # dstore = knn_dstore
                dstore.temp = self.args.interp_temp
                # TxBxC
                queries = net_output[1][self.args.knn_keytype]
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                yhat_knn_prob = dstore.get_knn_log_prob(
                        queries,
                        orig_target.permute(1, 0),
                        pad_idx=self.pad)
                yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    probs = probs.half()

                probs = combine_knn_and_vocab_probs(yhat_knn_prob, probs, self.args.lmbda)
                avg_probs = probs

            reps = reps.contiguous().view(-1, reps.shape[-1])
            labels = orig_target.view(-1)
            
            self.update_mem(reps[labels != self.pad], labels[labels != self.pad], sample['id'])

        avg_probs = avg_probs.view(sample['target'].shape)

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'dstore_keys': net_output[1][self.args.knn_keytype][start_idxs[i]:,i,:] if self.args.save_knnlm_dstore else None,
            }])
        return hypos

    def check_mem(self, sample):
        if self.last_ids is not None and sample['id'][0] != self.last_ids[-1] + 1:
            print("Memory doesn't match. Clear!")
            self.mem = None
            self.last_ids = None
            self.tot_mem = 0
    
    def update_mem(self, new_reps, new_labels, new_ids):
        if self.mem is not None:
            new_reps = torch.cat((self.mem[0], new_reps.detach()), dim=0)
            new_labels = torch.cat((self.mem[1], new_labels.detach()), dim=0)

        L = new_reps.shape[0]
        self.mem = (new_reps[L-self.args.mem_size:], new_labels[L-self.args.mem_size:])
        self.last_ids = new_ids
        self.tot_mem += new_reps.shape[0]

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
            combine_probs = torch.stack([vocab_p, knn_p], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - coeff)
            coeffs[1] = np.log(coeff)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

            return curr_prob
        
        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for i, (bd, tgt, is_single) in enumerate(batched):
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data

                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if 'knn_dstore' in kwargs:
                dstore = kwargs['knn_dstore']
                # TxBxC
                queries = bd[1][self.args.knn_keytype]
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                yhat_knn_prob = dstore.get_knn_log_prob(
                        queries,
                        orig_target.permute(1, 0),
                        pad_idx=self.pad)
                yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    probs = probs.half()

                probs = combine_knn_and_vocab_probs(
                            yhat_knn_prob, probs, self.args.lmbda)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'dstore_keys': decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:] if self.args.save_knnlm_dstore else None,
            }])
        return hypos
