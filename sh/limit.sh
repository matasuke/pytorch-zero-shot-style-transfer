# preprocess small parallel corpora
python preprocessor/limit_num_tokens_bi.py \
    --source_paths \
        'data/corpora/raw/OpenSubtitles.en-ja.en' \
        'data/corpora/raw/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/raw/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/raw/TED.20190427.en' \
        'data/corpora/raw/TED.ch' \
        'data/corpora/raw/ted-si.en' \
    --target_paths \
        'data/corpora/raw/OpenSubtitles.en-ja.ja' \
        'data/corpora/raw/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/raw/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/raw/TED.20190427.ja' \
        'data/corpora/raw/TED.ja' \
        'data/corpora/raw/ted-si.ja' \
    --save_dir data/corpora/processed/limit-50/ \
    --max_tokens 50
