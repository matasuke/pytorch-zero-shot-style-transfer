# preprocess small parallel corpora
# english file should be tokenized by tokenizer.perl
python preprocessor/sentence_piece_tokenizer.py \
    --data_paths \
        'data/corpora/processed/limit-50/sample-100000/OpenSubtitles.en-ja.en' \
        'data/corpora/processed/limit-50/sample-100000/OpenSubtitles.en-ja.ja' \
        'data/corpora/processed/limit-50/sample-100000/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/processed/limit-50/sample-100000/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/processed/limit-50/sample-100000/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/processed/limit-50/sample-100000/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/processed/limit-50/sample-100000/TED.20190427.en' \
        'data/corpora/processed/limit-50/sample-100000/TED.20190427.ja' \
        'data/corpora/processed/limit-50/sample-100000/TED.ch' \
        'data/corpora/processed/limit-50/sample-100000/TED.ja' \
     \
    --tokenizer_save_dir data/corpora/processed/limit-50/sample-100000/spm_model \
    --data_save_dir data/corpora/processed/limit-50/sample-100000/tokenized/ \
    --vocab_size 32000 \
    --model_type 'unigram' \
    --coverage 0.9995 \
    --sampling
