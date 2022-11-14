# This code is used when training the TrimeMT model
# This code is modified from label_smoothed_cross_entropy.py

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

import torch.distributed as dist
import torch.nn.functional as F

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    # Ignoring epsilon -- smoothed loss is not supported yet

    assert target.dim() == lprobs.dim()
    nll_loss = -lprobs
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()

    return nll_loss, nll_loss

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

def compute_sim(reps1, reps2, dist='ip'):
    if dist == 'l2':
        a2 = torch.sum(reps1 ** 2, dim=-1, keepdim=True)
        b2 = torch.sum(reps2 ** 2, dim=-1, keepdim=True)
        ip = torch.mm(reps1, reps2.T)
        d = a2 + b2.T - ip * 2
        sim = d * -1
    elif dist == 'ip':
        sim = torch.mm(reps1, reps2.T)
    else:
        raise NotImplementedError

    # scaled with sqrt(d)
    return sim / math.sqrt(reps1.shape[-1])


@register_criterion("trime_mt_loss")
class TrimeMTLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.dist_func = task.args.dist_func
        self.return_ce_loss = False
        self.temp = task.args.temp
        self.topk_mem = task.args.topk_mem

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--dist-func', default='ip', type=str, help='distance function used in Trime')
        # fmt: on


    def compute_in_bacth_logits(self, reps, labels, topk_mem=None, only_cross_sent=True):
        B = reps.shape[0]
        L = reps.shape[1]
        reps = reps.contiguous().view(-1, reps.shape[-1])
        labels = labels.contiguous().view(-1)

        inbatch_reps, local_range = my_gather(reps)
        inbatch_labels, _ = my_gather(labels)

        # remove padding from my_gather
        local_range_st = (inbatch_labels != 0)[:local_range[0]].sum()
        inbatch_reps = inbatch_reps[inbatch_labels != 0]
        inbatch_labels = inbatch_labels[inbatch_labels != 0]

        bsz = reps.shape[0]

        inbatch_logits = compute_sim(reps, inbatch_reps, self.dist_func) / self.temp

        local_mask = torch.ones((L, L), device=reps.device)
        if only_cross_sent:
            local_mask = local_mask * -10000.0
        else:
            local_mask = torch.triu(local_mask, diagonal=0) * -10000.0
        local_mask = torch.block_diag(*((local_mask, ) * B))
        assert(local_mask.shape[0] == bsz)

        inbatch_mask = torch.zeros(inbatch_logits.shape, device=reps.device).type_as(reps)
        inbatch_mask[:, local_range_st:local_range_st+bsz] = local_mask

        inbatch_logits = inbatch_logits + inbatch_mask

        # don't retrieve padding tokens
        inbatch_logits[:, inbatch_labels == self.padding_idx] -= 10000.0

        if topk_mem is not None:
            topk_logits = torch.topk(inbatch_logits, topk_mem, dim=-1)
            inbatch_logits = inbatch_logits.gather(dim=-1, index=topk_logits.indices)
            inbatch_labels = inbatch_labels[topk_logits.indices]

        return inbatch_logits, inbatch_labels


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        logits = net_output[0]
        norm_t = torch.logsumexp(logits, dim=-1).view(-1)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)

        lprobs = lprobs.view(-1, lprobs.shape[-1]).gather(dim=-1, index=target.view(-1).unsqueeze(-1)).view(-1)

        if not self.return_ce_loss:
            reps = net_output[1]['knn_embed']
            in_batch_logits, in_batch_labels = self.compute_in_bacth_logits(reps, target, topk_mem=self.topk_mem)

            in_batch_logs = F.log_softmax(in_batch_logits, dim=-1)
            negatives = (target.view(-1, 1) != (in_batch_labels.view(1, -1) if in_batch_labels.dim() == 1 else in_batch_labels))
            ctx_lprobs = torch.logsumexp(in_batch_logs + negatives * -10000.0, dim=-1)
    
            norm_c = torch.logsumexp(in_batch_logits, dim=-1)

            norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)
            norm_lprobs = torch.logsumexp(torch.stack((lprobs + norm_t - norm_tpc, ctx_lprobs + norm_c - norm_tpc), dim=-1), dim=-1)

            lprobs = norm_lprobs.view(target.shape)

        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
