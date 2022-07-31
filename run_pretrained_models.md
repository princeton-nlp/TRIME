# Run Pre-Trained Models
This file includes scripts to download and run pre-trained models. We assume that the dependencies have been installed and the datasets have been pre-processed.

## Table of all pre-trained models
| Dataset                         | Model        | Dev   | Test  | Hyper-parameters                                                            |
|---------------------------------|--------------|-------|-------|-----------------------------------------------------------------------------|
| Wikitext103 | [TrimeLM](https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-247M-trime.zip)<br/>(247M, L=3072)      | 17.10 | 17.76 | `--softmax-temp 1.17`                                                       |
| Wikitext103 | [TrimeLM_long](https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-247M-trime_long.zip)<br/>(247M, L=3072) | 17.01 | 17.64 | `--softmax-temp 1.22` `--mem-size 12288`                                    |
| Wikitext103 | [TrimeLM_ext](https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-247M-trime_ext.zip)<br/>(247M, L=3072) | 15.54 | 15.46 | `--softmax-temp 1.25` `--mem-size 12288` `--interp-temp 10.5` `--lmbda 0.3` |
| Wikitext103  | [TrimeLM](https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-150M-trime.zip)<br/>(150M, L=150)     | 24.45 | 25.61 | `--softmax-temp 1.03`                                                       |
| Wikitext103  | [TrimeLM_long](https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-150M-trime_long.zip)<br/>(150M, L=150) | 21.76 | 22.62 | `--softmax-temp 1.07` `--mem-size 15000`                                    |
| enwik8       | [TrimeLM](https://nlp.cs.princeton.edu/projects/trime/pretrained_models/enwik8-38M-trime.zip)<br/>(38M, L=512)     | 1.14  | 1.12  | `--softmax-temp 1.05`                                                       |
| enwik8       | [TrimeLM_long](https://nlp.cs.princeton.edu/projects/trime/pretrained_models/enwik8-38M-trime_long.zip)<br/>(38M, L=512) | 1.08  | 1.05  | `--softmax-temp 1.10` `--mem-size 24576`                                    |


## Wikitext-103 (247M, L=3072)

### TrimeLM (local memory)
TrimeLM uses only the local memory (constructed using tokens in the input). It can be viewed as a lightweight replacement for vanilla langauge models.
```bash
# download the pre-trained TrimeLM
mkdir pretrained_models; cd pretrained_models
wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-247M-trime.zip;
unzip wiki103-247M-trime.zip; rm -f wiki103-247M-trime.zip
cd ..

# run evaluation
python eval_lm-trime.py data-bin/wikitext-103 \
    --path pretrained_models/wiki103-247M-trime/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 --context-window 2560 \
    --softmax-batch 1024 --gen-subset valid --fp16 \
    --max-sentences 1 --knn-keytype last_ffn_input \
    --use-local --softmax-temp 1.17

# the following output is expected:
# Loss (base 2): 4.0962, Perplexity: 17.10
```
Arguments:
* `--use-local` specifies using local memory.
* `--softmax-temp` specifies the temperature term used when computing the loss.

### TrimeLM_long (local + long-term memory)
TrimeLM_long uses local memory and long-term memory during inference. The model is able to leverage long contexts, although it is trained with shorter ones.
```bash
# download the pre-trained TRIME_long
mkdir pretrained_models; cd pretrained_models
wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-247M-trime_long.zip;
unzip wiki103-247M-trime_long.zip; rm -f wiki103-247M-trime_long.zip
cd ..

# run evaluation
python eval_lm-trime.py data-bin/wikitext-103 \
    --path pretrained_models/wiki103-247M-trime_long/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 --context-window 2560 \
    --softmax-batch 1024 --gen-subset valid --fp16 \
    --max-sentences 1 --knn-keytype last_ffn_input \
    --use-local --use-long --mem-size 12288 --softmax-temp 1.22

# the following output is expected:
# Loss (base 2): 4.0879, Perplexity: 17.01
```
Arguments:
* `--use-long` specifies using long-term memory.
* `--mem-size` specifies the size of local + long-term memory.

### TrimeLM_ext (local + long-term + external memory)
TrimeLM_ext uses local memory, long-term memory, and external memory. During inference, we run the model on the training set to build the external memory and use Faiss library to build index for retrieving top-K nearest neighbors the external memory. We also calibrate a separated distribution over the memory and interpolate the output distribution and the memory distribution, similarly to kNN-LM (see details in the paper).

We first download the pre-trained TrimeLM_ext:
```bash
mkdir pretrained_models; cd pretrained_models
wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-247M-trime_ext.zip;
unzip wiki103-247M-trime_ext.zip; rm -f wiki103-247M-trime_ext.zip
cd ..
```

Then, we generate the external memory (keys and values) using the training set and then build the Faiss index:
```bash
MODEL_PATH=pretrained_models/wiki103-247M-trime_ext

# generate the external memory (keys and values) using the training set
python eval_lm.py data-bin/wikitext-103 \
    --path ${MODEL_PATH}/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 2560 --tokens-per-sample 512 \
    --dstore-mmap ${MODEL_PATH}/dstore --knn-keytype last_ffn_input \
    --dstore-size 103224461 \
    --save-knnlm-dstore --fp16 --dstore-fp16


# build Faiss index
python build_dstore.py \
    --dstore_mmap ${MODEL_PATH}/dstore \
    --dstore_size 103224461 --dimension 1024 \
    --faiss_index ${MODEL_PATH}/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0  --dstore_fp16  --dist ip
```

Now, we are ready to evaluate the model:
```bash
MODEL_PATH=pretrained_models/wiki103-247M-trime_ext

python eval_lm-trime.py data-bin/wikitext-103 \
    --path ${MODEL_PATH}/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 --context-window 2560 \
    --softmax-batch 1024 --gen-subset valid --fp16 \
    --max-sentences 1 --knn-keytype last_ffn_input \
    --use-local --use-long --mem-size 12288 --softmax-temp 1.25 \
    --use-external --dstore-filename ${MODEL_PATH}/dstore --indexfile ${MODEL_PATH}/knn.index.ip \
    --probe 32 --dstore-fp16 --faiss-metric-type ip --no-load-keys --k 1024 \
    --use-interp --interp-temp 10.5 --lmbda 0.3 

# the following output is expected:
# Loss (base 2): 3.9580, Perplexity: 15.54
```
Arguments: 
* `--use-external` specifies using external memory.
* `--dstore-filename` and `indexfile` specify the datastore and the Faiss index paths.
* `--use-interp` specifies using a linear interpolation between two distributions to calibrate final probablity.
* `--lmbda` and `--interp-temp` specify the temerpature term and the weight when using the linear interpolation.

## Wikitext-103 (150M, L=150)

### TrimeLM (local memory)
```bash
# download the pre-trained TrimeLM
mkdir pretrained_models; cd pretrained_models
wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-150M-trime.zip;
unzip wiki103-150M-trime.zip; rm -f wiki103-150M-trime.zip
cd ..

# run evaluation
python eval_lm-trime.py data-bin/wikitext-103 \
    --path pretrained_models/wiki103-150M-trime/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 150 --context-window 86 \
    --softmax-batch 1024 --gen-subset valid --fp16 \
    --max-sentences 1 --knn-keytype last_ffn_input \
    --use-local --softmax-temp 1.03

# the following output is expected:
# Loss (base 2): 4.6119, Perplexity: 24.45
```

### TrimeLM_long (local + long-term memory)
```bash
# download the pre-trained TrimeLM
mkdir pretrained_models; cd pretrained_models
wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/wiki103-150M-trime_long.zip;
unzip wiki103-150M-trime_long.zip; rm -f wiki103-150M-trime_long.zip
cd ..

# run evaluation
python eval_lm-trime.py data-bin/wikitext-103 \
    --path pretrained_models/wiki103-150M-trime_long/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 150 --context-window 86 \
    --softmax-batch 1024 --gen-subset valid --fp16 \
    --max-sentences 1 --knn-keytype last_ffn_input \
    --use-local --use-long --mem-size 15000 --softmax-temp 1.07

# the following output is expected:
# Loss (base 2): 4.4433, Perplexity: 21.76
```

## Enwik8 (38M, L=512)

### TrimeLM (local memory)
```bash
# download the pre-trained TrimeLM
mkdir pretrained_models; cd pretrained_models
wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/enwik8-38M-trime.zip;
unzip enwik8-38M-trime.zip; rm -f enwik8-38M-trime.zip
cd ..

# run evaluation
python eval_lm-trime.py data-bin/enwik8 \
    --path pretrained_models/enwik8-38M-trime/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 512 --context-window 432 \
    --softmax-batch 1024 --gen-subset valid --fp16 \
    --max-sentences 1 --knn-keytype last_ffn_input \
    --use-local --softmax-temp 1.05

# the following output is expected:
# Loss (base 2): 1.1411, Perplexity: 2.21
```

### TrimeLM_long (local + long-term memory)
```bash
# download the pre-trained TrimeLM
mkdir pretrained_models; cd pretrained_models
wget https://nlp.cs.princeton.edu/projects/trime/pretrained_models/enwik8-38M-trime_long.zip;
unzip enwik8-38M-trime_long.zip; rm -f enwik8-38M-trime_long.zip
cd ..

# run evaluation
python eval_lm-trime.py data-bin/enwik8 \
    --path pretrained_models/enwik8-38M-trime_long/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 512 --context-window 432 \
    --softmax-batch 1024 --gen-subset valid --fp16 \
    --max-sentences 1 --knn-keytype last_ffn_input \
    --use-local --use-long --mem-size 24576 --softmax-temp 1.10

# the following output is expected:
# Loss (base 2): 1.0802, Perplexity: 2.11
```
