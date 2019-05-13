# preprocess small parallel corpora
python preprocessor/sampler.py \
    --source_paths \
        'data/corpora/raw/OpenSubtitles.en-ja.en' \
        'data/corpora/raw/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/raw/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/raw/ted-si.en' \
        'data/corpora/raw/TED.ch' \
     \
    --target_paths \
        'data/corpora/raw/OpenSubtitles.en-ja.ja' \
        'data/corpora/raw/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/raw/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/raw/ted-si.ja' \
        'data/corpora/raw/TED.ja' \
    --save_dir data/corpora/sample-50000 \
    --sample_size 50000
