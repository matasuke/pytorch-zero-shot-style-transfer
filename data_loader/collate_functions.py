from typing import List, Tuple

import torch


def seq2seq_collate_fn(
    inputs: List[Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, ...]:
    '''
    create mini-batch tensors from source target sentences.
    use this collate_fn to pad sentences.

    :param inputs: mini batch of source and target sentences with languages and styles.

    NOTE
    ----
    inputs contains batch o src sentences, tgt sentences, languages, styles like
    (source_sentences, target_sentences, target_languages, target_styles)
    '''
    def merge_seq(sentences: List[torch.Tensor]):
        '''
        pad sequences for source
        '''
        lengths = [len(sen) for sen in sentences]
        padded_seqs = torch.zeros(len(sentences), max(lengths)).long()

        for idx, sen in enumerate(sentences):
            end = lengths[idx]
            padded_seqs[idx, :end] = sen[:end]

        padded_seqs = padded_seqs.t().contiguous()

        return padded_seqs, lengths

    def merge_features(
            features: List[torch.Tensor],
            lengths: List[int]
    ) -> torch.Tensor:
        '''
        pad features like language and style
        convert List[batch_size] -> [seq_len, batch_size]
        '''
        padded_feats = torch.zeros(len(features), max(lengths)).long()

        for idx, feat in enumerate(features):
            end = lengths[idx]
            padded_feats[idx, :end] = feat

        padded_feats = padded_feats.t().contiguous()

        return padded_feats

    indices = list(range(len(inputs)))

    # sort a list of sentence length based on source sentence to use pad_padded_sequence
    src_tgt_pair, indices = \
        zip(*sorted(zip(inputs, indices), key=lambda x: len(x[0][0]), reverse=True))
    src, tgt, tgt_lang, tgt_style = zip(*src_tgt_pair)

    src, lengths = merge_seq(src)
    tgt, _ = merge_seq(tgt)

    tgt_lang = merge_features(tgt_lang, lengths)
    tgt_style = merge_features(tgt_style, lengths)

    lengths = torch.LongTensor([lengths])

    return src, tgt, tgt_lang, tgt_style, lengths, indices
