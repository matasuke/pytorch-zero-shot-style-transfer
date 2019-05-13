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
    def merge(sentences: List[torch.Tensor]):
        '''
        pad sequences for source
        '''
        lengths = [len(sen) for sen in sentences]
        padded_seqs = torch.zeros(len(sentences), max(lengths)).long()

        for idx, sen in enumerate(sentences):
            end = lengths[idx]
            padded_seqs[idx, :end] = sen[:end]

        lengths = torch.LongTensor([lengths])
        padded_seqs = padded_seqs.t().contiguous()

        return padded_seqs, lengths

    indices = list(range(len(inputs)))

    # sort a list of sentence length based on source sentence to use pad_padded_sequence
    src_tgt_pair, indices = \
        zip(*sorted(zip(inputs, indices), key=lambda x: len(x[0][0]), reverse=True))
    src, tgt, tgt_lang, tgt_style = zip(*src_tgt_pair)

    src, lengths = merge(src)
    tgt, _ = merge(tgt)

    # tgt_lang and tgt_style: [seq_len, batch_size, lang_dim], [seq_len, batch_size, style_dim]
    # embedding of lang and style are concatenated to vocab_dim
    tgt_lang = torch.stack([torch.stack(tgt_lang, -1) for _ in range(0, src.size(0))]).squeeze(1)
    tgt_style = torch.stack([torch.stack(tgt_style, -1) for _ in range(0, src.size(0))]).squeeze(1)

    return src, tgt, tgt_lang, tgt_style, lengths, indices
