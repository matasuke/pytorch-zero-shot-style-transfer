# preprocess small parallel corpora
python preprocessor/text_preprocessor.py \
    --data_paths \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/OpenSubtitles.en-ja.en' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/OpenSubtitles.en-ja.ja' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/TED.20190427.en' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/TED.20190427.ja' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/TED.ch' \
        'data/corpora/processed/limit-50/sample-100000/tokenized/tokens/TED.ja' \
     \
    --languages '<2EN>' '<2JA>' '<2CH>' \
    --styles '<2OST>' '<2SI>' '<2TED>' \
    --save_path data/corpora/processed/limit-50/sample-100000/tokenized/tokens/preprocessor/ost-ted-si.32000.pkl \
    --max_vocab_size 32000
