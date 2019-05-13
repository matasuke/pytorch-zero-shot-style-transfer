# preprocess small parallel corpora
python preprocessor/text_preprocessor.py \
    --data_paths \
        'data/corpora/tokenized/OpenSubtitles.ja-zh_cn.tokenized.ja' \
        'data/corpora/tokenized/OpenSubtitles.ja-zh_cn.tokenized.zh_cn' \
        'data/corpora/tokenized/ted-si.tokenized.en' \
        'data/corpora/tokenized/ted-si.tokenized.ja' \
        'data/corpora/tokenized/TED.tokenized.ch' \
        'data/corpora/tokenized/TED.tokenized.ja' \
     \
    --languages '<2EN>' '<2JA>' '<2CH>' \
    --styles '<2OST>', '<2SI>' '<2TED>' \
    --save_path data/preprocessor/test_ted-si.32000.pkl \
    --max_vocab_size 32000
