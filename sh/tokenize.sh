# preprocess small parallel corpora
python preprocessor/sentence_piece_tokenizer.py \
    --data_paths \
        'data/corpora/sample-50000/OpenSubtitles.en-ja.en' \
        'data/corpora/sample-50000/OpenSubtitles.en-ja.ja' \
        'data/corpora/sample-50000/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/sample-50000/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/sample-50000/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/sample-50000/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/sample-50000/ted-si.en' \
        'data/corpora/sample-50000/ted-si.ja' \
        'data/corpora/sample-50000/TED.ch' \
        'data/corpora/sample-50000/TED.ja' \
     \
    --tokenizer_save_dir data/corpora/sample-50000/spm_model \
    --data_save_dir data/corpora/sample-50000/tokenized/ \
    --vocab_size 32000 \
    --model_type 'unigram' \
    --coverage 0.9995 \
    --sampling
