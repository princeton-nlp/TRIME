import math

import torch.distributed as dist
import torch.nn.functional as F
import torch

import random

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def my_gather(t):
    if dist.is_initialized():
        local_size = torch.tensor([t.shape[0]], dtype=torch.int64, device=t.device)
        size_list = [torch.tensor([0], dtype=torch.int64, device=t.device) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=size_list, tensor=local_size)
        max_size = max([int(size.cpu().item()) for size in size_list])

        if local_size != max_size:
            t_shape = list(t.shape)
            t_shape[0] = max_size
            tmp_t = torch.zeros(t_shape, dtype=t.dtype, device=t.device)
            tmp_t[:local_size] = t
            t = tmp_t

        t_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=t_list, tensor=t.contiguous())
        t_list[dist.get_rank()] = t
        t = torch.cat(t_list, 0)
        range_l = dist.get_rank() * max_size
        range_r = (dist.get_rank() + 1) * max_size
        return t, (range_l, range_r)
    else:
        return t, (0, t.shape[0])

def compute_in_bacth_logits(reps, labels, mem_size, num_docs, only_cross_sent=False):
    B = reps.shape[0]
    L = reps.shape[1]
    reps = reps.contiguous().view(-1, reps.shape[-1])
    # labels = labels.contiguous().view(-1)

    inbatch_reps, local_range = my_gather(reps)
    inbatch_labels, _ = my_gather(labels)

    # mask: only attend to tokens in previous segments and previous token in the same segment
    local_mask = torch.ones((mem_size, mem_size), device=reps.device)
    local_mask = torch.tril(local_mask, diagonal=-1)
    local_mask = 1 - torch.block_diag(*((local_mask, ) * num_docs))
    assert local_mask.shape[0] == inbatch_reps.shape[0]
    local_mask = local_mask * -10000.0
    local_mask = local_mask[inbatch_labels != 0]
    local_mask = local_mask[:, inbatch_labels != 0]

    # remove padding from my_gather
    local_range_st = (inbatch_labels != 0)[:local_range[0]].sum()
    inbatch_reps = inbatch_reps[inbatch_labels != 0]
    inbatch_labels = inbatch_labels[inbatch_labels != 0]

    bsz = reps.shape[0]

    inbatch_logits = torch.mm(reps, inbatch_reps.T) / math.sqrt(reps.shape[-1])

    # mask: only attend to previous tokens
    inbatch_mask = local_mask[local_range_st:local_range_st+bsz, :]

    inbatch_logits = inbatch_logits + inbatch_mask

    return inbatch_logits, inbatch_labels


@register_criterion('trime_long_loss')
class TrimeLongLoss(FairseqCriterion):
    """
    This is an implementation of the Trime loss. 
    In this function, we batch consecutive segments in one batch and use tokens from previous segments to construct memory.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        self.dist = args.dist
        self.temp = args.temp

        self.return_ce_loss = False
        self.mem_size = args.max_tokens * args.num_comb_shards
        self.num_docs = args.distributed_world_size // args.num_comb_shards

        if args.ddp_backend == 'c10d':
            raise Exception(
                'AdaptiveLossConloss is not compatible with the c10d '
                'version of DistributedDataParallel. Please use '
                '`--ddp-backend=no_c10d` instead.'
            )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model.decoder, 'adaptive_softmax') and model.decoder.adaptive_softmax is not None
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample['net_input'])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target, target_idxs = adaptive_softmax(net_output[0], orig_target, return_target_idx=True)
        assert len(target) == len(logits)
        norm_t = torch.logsumexp(logits[0], dim=-1)

        token_loss = F.cross_entropy(logits[0], target[0], ignore_index=self.padding_idx, reduction='none')
        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                token_loss[target_idxs[i]] += F.cross_entropy(logits[i + 1], target[i + 1], ignore_index=self.padding_idx, reduction='none')

        ori_loss = token_loss.sum(-1).view(-1)

        if self.return_ce_loss:
            loss = ori_loss
            orig = utils.strip_pad(orig_target, self.padding_idx)
            ntokens = orig.numel()
            sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
            logging_output = {
                'loss': loss.data.double(),
                'ori_loss': ori_loss.data.double(),
                'norm_loss': 0.0,
                'ntokens': ntokens,
                'nsentences': nsentences,
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output

        # reps: B x L x d
        # rep_labels: B x L
        reps = net_output[1][self.args.knn_keytype].permute(1, 0, 2)

        # in_batch_logits: (B x L) x I (I = in-batch examples)
        if self.training and (self.args.cross_sent_ratio is not None):
            in_batch_logits, in_batch_labels = compute_in_bacth_logits(reps, orig_target, 
                        mem_size=self.mem_size, num_docs=self.num_docs, only_cross_sent=(random.random() < self.args.cross_sent_ratio))
        else:
            in_batch_logits, in_batch_labels = compute_in_bacth_logits(reps, orig_target, mem_size=self.mem_size, num_docs=self.num_docs)

        norm_c = torch.logsumexp(in_batch_logits, dim=-1)

        in_batch_logs = F.log_softmax(in_batch_logits, dim=-1)
        negatives = (orig_target.view(-1, 1) != (in_batch_labels.view(1, -1) if in_batch_labels.dim() == 1 else in_batch_labels))
        ctx_loss = -torch.logsumexp(in_batch_logs + negatives * -10000.0, dim=-1)

        # normalize token loss and ctx loss
        norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)
        norm_loss = -torch.logsumexp(torch.stack((-token_loss + norm_t - norm_tpc, -ctx_loss + norm_c - norm_tpc), dim=-1), dim=-1)
        norm_loss = norm_loss.sum(-1).view(-1)

        loss = norm_loss

        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
        logging_output = {
            'loss': loss.data.double(),
            'ori_loss': ori_loss.data.double(),
            'norm_loss': norm_loss.data.double(),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ori_loss_sum = utils.item(sum(log.get('ori_loss', 0) for log in logging_outputs))
        norm_loss_sum = utils.item(sum(log.get('norm_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        if sample_size == 0:
            metrics.log_scalar('loss', 0.0, 0, round=3)
            metrics.log_scalar('ori_loss', 0.0, 0, round=3)
            metrics.log_scalar('norm_loss', 0.0, 0, round=3)
            return

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ori_loss', ori_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('norm_loss', norm_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
