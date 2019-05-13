# preprocess small parallel corpora
python preprocessor/text_preprocessor.py \
    --data_paths \
        'data/corpora/sample-50000/tokenized/limit/OpenSubtitles.en-ja.en' \
        'data/corpora/sample-50000/tokenized/limit/OpenSubtitles.en-ja.ja' \
        'data/corpora/sample-50000/tokenized/limit/OpenSubtitles.en-zh_cn.en' \
        'data/corpora/sample-50000/tokenized/limit/OpenSubtitles.en-zh_cn.zh_cn' \
        'data/corpora/sample-50000/tokenized/limit/OpenSubtitles.ja-zh_cn.ja' \
        'data/corpora/sample-50000/tokenized/limit/OpenSubtitles.ja-zh_cn.zh_cn' \
        'data/corpora/sample-50000/tokenized/limit/ted-si.en' \
        'data/corpora/sample-50000/tokenized/limit/ted-si.ja' \
        'data/corpora/sample-50000/tokenized/limit/TED.ch' \
        'data/corpora/sample-50000/tokenized/limit/TED.ja' \
     \
    --languages '<2EN>' '<2JA>' '<2CH>' \
    --styles '<2OST>' '<2SI>' '<2TED>' \
    --save_path data/corpora/sample-50000/preprocessor/ost-ted-si.32000.pkl \
    --max_vocab_size 32000
