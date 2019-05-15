# preprocess small parallel corpora
python preprocessor/sampler.py \
    --source_paths \
        'data/corpora/processed/limit-50/OpenSubtitles.en-ja.en' \
        'data/corpora/processed/limit-50/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/processed/limit-50/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/processed/limit-50/TED.20190427.en' \
        'data/corpora/processed/limit-50/TED.ch' \
        'data/corpora/processed/limit-50/ted-si.en' \
     \
    --target_paths \
        'data/corpora/processed/limit-50/OpenSubtitles.en-ja.ja' \
        'data/corpora/processed/limit-50/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/processed/limit-50/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/processed/limit-50/TED.20190427.ja' \
        'data/corpora/processed/limit-50/TED.ja' \
        'data/corpora/processed/limit-50/ted-si.ja' \
    --save_dir data/corpora/processed/limit-50/sample-100000/ \
    --sample_size 100000
