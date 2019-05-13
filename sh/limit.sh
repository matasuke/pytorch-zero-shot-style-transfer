# preprocess small parallel corpora
python preprocessor/limit_num_tokens_bi.py \
    --source_paths \
        'data/corpora/sample-50000/tokenized/OpenSubtitles.en-ja.en' \
        'data/corpora/sample-50000/tokenized/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/sample-50000/tokenized/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/sample-50000/tokenized/ted-si.en' \
        'data/corpora/sample-50000/tokenized/TED.ch' \
     \
    --target_paths \
        'data/corpora/sample-50000/tokenized/OpenSubtitles.en-ja.ja' \
        'data/corpora/sample-50000/tokenized/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/sample-50000/tokenized/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/sample-50000/tokenized/ted-si.ja' \
        'data/corpora/sample-50000/tokenized/TED.ja' \
    --save_dir data/corpora/sample-50000/tokenized/limit \
    --max_tokens 50
