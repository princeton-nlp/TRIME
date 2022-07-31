# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('linear')
class LinearSchedule(FairseqLRScheduler):
    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with linear.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        self.base_lr = args.lr[0]
        self.tot_steps = args.max_update
        self.warmup_steps = max(0, int(self.tot_steps * args.warmup_ratio))
        self.optimizer.set_lr(self.base_lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-ratio', default=0.1, type=float, metavar='r',
                            help='warmup the learning rate linearly for the first r*T updates')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_steps:
            self.lr = self.base_lr * (float(num_updates + 1) / float(max(1, (self.warmup_steps + 1))))
        else:
            self.lr = self.base_lr * (float(self.tot_steps - num_updates) / float(max(1, self.tot_steps - self.warmup_steps)))
        self.optimizer.set_lr(self.lr)
        return self.lr
