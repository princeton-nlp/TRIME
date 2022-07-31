#!/bin/sh

python train.py --task language_modeling data-bin/wikitext-103 \
    --save-dir output/wiki103-247M-trime_ext \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion trime_ext_loss --max-tokens 3072 --update-freq 6 --tokens-per-sample 3072 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --knn-keytype last_ffn_input --fp16 \
    --ce-warmup-epoch 9 --cross-sent-ratio 0.9 \
    --predefined-batches data-bin/wikitext-103/wiki103-l3072-batches.json