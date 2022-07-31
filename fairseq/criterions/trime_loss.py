import math

import torch.distributed as dist
import torch.nn.functional as F
import torch

import random

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def compute_in_bacth_logits(reps, labels):
    B = reps.shape[0]
    L = reps.shape[1]
    reps = reps.contiguous().view(-1, reps.shape[-1])

    bsz = reps.shape[0]

    # compute scaled IP
    inbatch_logits = torch.mm(reps, reps.T) / math.sqrt(reps.shape[-1])
    inbatch_labels = labels

    # each token can only access previous tokens
    local_mask = torch.ones((L, L), device=reps.device)
    local_mask = torch.triu(local_mask, diagonal=0) * -10000.0
    local_mask[local_mask == 0] = 1
    local_mask = torch.block_diag(*((local_mask, ) * B))
    local_mask[local_mask == 0] = -10000.0
    local_mask[local_mask == 1] = 0
    assert(local_mask.shape[0] == bsz)

    inbatch_mask = local_mask

    inbatch_logits = inbatch_logits + inbatch_mask

    return inbatch_logits, inbatch_labels


@register_criterion('trime_loss')
class TrimeLoss(FairseqCriterion):
    """
    This is an implementation of the Trime loss. 
    In this function, only local memory will be used to compute the loss.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        self.dist = args.dist
        self.temp = args.temp

        self.return_ce_loss = False

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
        in_batch_logits, in_batch_labels = compute_in_bacth_logits(reps, orig_target)
        
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
