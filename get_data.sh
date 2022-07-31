dataset=$1

if [ "${dataset}" = "wikitext-103" ]; then
    echo Download and process wikitext-103

    mkdir -p data-bin/wikitext-103
    cd data-bin/wikitext-103

    mkdir raw_data
    cd raw_data
    wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    unzip wikitext-103-v1.zip

    cd ../../..
    python preprocess.py \
        --only-source \
        --trainpref data-bin/wikitext-103/raw_data/wikitext-103/wiki.train.tokens \
        --validpref data-bin/wikitext-103/raw_data/wikitext-103/wiki.valid.tokens \
        --testpref data-bin/wikitext-103/raw_data/wikitext-103/wiki.test.tokens \
        --destdir data-bin/wikitext-103 \
        --workers 20

elif [ "${dataset}" = "enwik8" ]; then
    echo Download and process enwik8

    mkdir -p data-bin/enwik8
    cd data-bin/enwik8

    mkdir raw_data
    cd raw_data
    wget --continue http://mattmahoney.net/dc/enwik8.zip
    wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
    python prep_enwik8.py

    cd ../../..
    python preprocess.py \
        --only-source \
        --trainpref data-bin/enwik8/raw_data/train.txt \
        --validpref data-bin/enwik8/raw_data/valid.txt \
        --testpref data-bin/enwik8/raw_data/test.txt \
        --destdir data-bin/enwik8 \
        --workers 20
else
    echo "Dataset ${dataset} is not supported!"
fi
