# Machine Translation with TrimeMT

This is the repository for the paper [Training Language Models with Memory Augmentation](https://arxiv.org/abs/2205.12674), by [Zexuan Zhong](https://www.cs.princeton.edu/~zzhong/), [Tao Lei](https://taolei87.github.io), and [Danqi Chen](https://www.cs.princeton.edu/~danqic/).

This folder includes the code for our machine translation experiments. 

Please find more details of this work in our [paper](https://arxiv.org/pdf/2205.12674.pdf).


## Preprocess Datasets
We conduct machine translation experiments on the IWSLT'14 German to English dataset. We use the following instructions to download and preprocess the data

```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
PYTHONPATH=. python fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

## Run Pre-Trained Models
We release a pre-trained TrimeMT model, and a pre-trained transformer enc-dec model as the vanilla baseline and kNN-MT baseline. You can use the following commands to download pre-trained models.
```bash
mkdir pretrained_models; cd pretrained_models

wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/iwslt14-vanilla.zip
unzip iwslt14-vanilla.zip; rm -f iwslt14-vanilla.zip

wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/iwslt14-trime.zip
unzip iwslt14-trime.zip; rm -f iwslt14-trime.zip
```

### Baseline: Transformer Encoder-Decoder
The vanilla transformer encoder-decoder model can be evaluated as follows.
```bash
MODEL_PATH=pretrained_models/iwslt14-vanilla

PYTHONPATH=. python fairseq_cli/generate.py data-bin/iwslt14.tokenized.de-en/ \
    --gen-subset test \
    --path $MODEL_PATH/checkpoint_best.pt \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --max-tokens 4096 \
    --tokenizer moses --remove-bpe --quiet

# BLEU = 32.58
```

### Baseline: kNN-MT
We evaluate the kNN-MT model as follows. The kNN-MT model is trained with the original training objective and evaluated with an external memory (we include the datastore and the Faiss index in the pre-trained models' folders; see how to build external memory below). 

```bash
MODEL_PATH=pretrained_models/iwslt14-vanilla
DSTORE_PATH=${MODEL_PATH}/dstore
DSTORE_SIZE=3949114
INDEX_FILENAME=knn_index_l2

k=32
l=0.2
t=100

PYTHONPATH=. python experimental_generate.py data-bin/iwslt14.tokenized.de-en/ \
    --gen-subset test \
    --path $MODEL_PATH/checkpoint_best.pt \
    --arch transformer_iwslt_de_en \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --max-tokens 4096 \
    --tokenizer moses --remove-bpe  --quiet \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DSTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': $k, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'fix', 'knn_lambda_value': $l, 'knn_temperature_type': 'fix', 'knn_temperature_value': $t,
    'index_filename': '$INDEX_FILENAME'}"

# BLEU = 33.15
```

Note: we tune the hyper-parameters (i.e., `k`, `l`, `t`) on the development set; we use L2 distance during inference and we find using inner product produces similar results.

### Our Approach: TrimeMT
Our model TrimeMT is trained with our proposed Trime training objective. During inference, we build an external memory using the training set (we include the datastore and the Faiss index in the pre-trained models' folders; see how to build external memory below).  We evaluate TrimeMT as follows.

```bash
MODEL_PATH=pretrained_models/iwslt14-trime
DSTORE_PATH=${MODEL_PATH}/dstore
DSTORE_SIZE=3949114
INDEX_FILENAME=knn_index_ip

k=16
t=15

PYTHONPATH=. python experimental_generate.py data-bin/iwslt14.tokenized.de-en/ \
    --gen-subset test \
    --path $MODEL_PATH/checkpoint_best.pt \
    --arch transformer_iwslt_de_en \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --max-tokens 4096 \
    --tokenizer moses --remove-bpe --quiet \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DSTORE_PATH', 'index_filename': 'knn_index_ip', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': $k, 'probe': 32,
    'knn_sim_func': None, 'faiss_metric_type': 'ip', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'trime', 'knn_lambda_value': 0.0, 'knn_temperature_type': 'fix', 'knn_temperature_value': $t,
    'index_filename': '$INDEX_FILENAME'}"

# BLEU = 33.73
```

Note: we tune the hyper-parameters (i.e., `k`, `t`) on the development set.

### Build External Memory
We generate the external memory (a datastore of keys and values) using the training set and then build the Faiss index. (We include the built datastore and the Faiss index in pre-trained models already; there is no need to re-build them.)
```bash
MODEL_PATH=pretrained_models/iwslt14-trime
DSTORE_SIZE=3949114
DSTORE_PATH=${MODEL_PATH}/dstore

CUDA_VISIBLE_DEVICES=0 python save_datastore.py data-bin/iwslt14.tokenized.de-en/ \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH/checkpoint_best.pt \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 512 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DSTORE_PATH

DIST=ip # we use DIST=l2 for kNN-MT

CUDA_VISIBLE_DEVICES=0 python train_datastore_gpu.py \
  --dstore_mmap $DSTORE_PATH \
  --dstore_size $DSTORE_SIZE \
  --dstore-fp16 \
  --faiss_index ${DSTORE_PATH}/knn_index_${DIST} \
  --ncentroids 4096 \
  --probe 32 \
  --dimension 512 --dist ${DIST}
```

## Train TrimeMT

### Training Scripts
We use the following commands to train our models. We use BM25 to group examples into training batches. You can download the BM25 batching results as the commands do. The BM25 batching scritps can also be found below.

Our models are trained on 4 NVIDIA RTX3090 GPUs.

```bash
# Download BM25 batching results
wget https://nlp.cs.princeton.edu/projects/trime/bm25_batch/iwslt14-batches.json -P data-bin/iwslt14.tokenized.de-en/
mv data-bin/iwslt14.tokenized.de-en/iwslt14-batches.json data-bin/iwslt14.tokenized.de-en/bn25_batches.json

# Train TrimeMT
PYTHONPATH=. python train.py data-bin/iwslt14.tokenized.de-en \
    --save-dir outputs/iwslt-trime \
    --arch transformer_iwslt_de_en \
    --share-decoder-input-output-embed --decoder-normalize-before \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --ce-warmup-epoch 5 --dropout 0.3 --weight-decay 0.0001 \
    --criterion trime_mt_loss \
    --label-smoothing 0.0 \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --predefined-batches data-bin/iwslt14.tokenized.de-en/bm25_batches.json \
    --max-epoch 50 --dist-func ip \
    --temp 10.0 --topk-mem 8
```
Arguments:
* `--criterion` specifies the training objective we used during training. Our proposed objective can be specified by `--criterion trime_mt_loss`; the vanilla objective can be specified by `--criterion label_smoothed_cross_entropy`
* `--predefined-batches` specifies the file path of the predefined batches (we use BM25 to batch segments).
* `--temp` specifies the temperature we used during training.
* `--topk-mem` specifies how many in-batch tokens are used to calibrate the prediction during training. In machine translation experiments, we only take top-K nearest neighbors among in-batch tokens to compute the distribution. 

### BM25 Batching
We use BM25 scores to batch training examples, when we train the TrimeMT model.

We use the [Pyserini](https://github.com/castorini/pyserini) library to build BM25 index. The library can be installed via pip.
```bash
pip install pyserini
```

We first save all the examples (only in the target language) from the training set into a .json file.
```bash
mkdir data-bin/iwslt14.tokenized.de-en/blocks/
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train.py data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --optimizer adam --reset-optimizer \
    --output-samples-to-file data-bin/iwslt14.tokenized.de-en/blocks/train_samples.json
```

Then, we build the BM25 index using Pyserini.
```bash
python -m pyserini.index.lucene --collection JsonCollection --input data-bin/iwslt14.tokenized.de-en/blocks/ --index data-bin/iwslt14.tokenized.de-en/bm25_index --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw
```

Next, for each training segment, we search the similar segments using the BM25 index we built above.

```bash
python bm25_search.py \
    --index_path data-bin/iwslt14.tokenized.de-en/bm25_index/ \
    --samples_path data-bin/iwslt14.tokenized.de-en/blocks/train_samples.json \
    --results_path data-bin/iwslt14.tokenized.de-en/bm25_results
```

Finally, based on the retrieval results, we create batches by group similar segments.
```bash
python bm25_make_batches.py \
    --results_path data-bin/iwslt14.tokenized.de-en/bm25_results \
    --batch_file data-bin/iwslt14.tokenized.de-en/bm25_batches.json
```
The resulting file `data-bin/iwslt14.tokenized.de-en/bm25_batches.json` can be used as `--predefined-batches` when training TrimeMT.

# Acknowledgement
This codebase is based on the [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) project. We thank the authors for open-sourcing the great code!
