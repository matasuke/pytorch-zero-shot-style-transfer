'''
delete sentences which have more than specified number of tokens.
'''
import argparse
from typing import List, Tuple
from pathlib import Path

from tqdm import tqdm


def limit_num_tokens_parallel(
        source_sentences: List[str],
        target_sentences: List[str],
        max_tokens: int = 50,
) -> Tuple[List[str], List[str]]:
    '''
    Limit the number of tokens in input file.
    sentences which has more than token_limit tokens is discarded.
    sentences has to be pre-tokenized.

    :param source_sentences: list of source sentences.
    :param target_sentences: list of target sentences.
    :param max_tokens: max number of tokens.
    '''

    assert max_tokens >= 1
    assert len(source_sentences) == len(target_sentences)

    exclude_idx_list = []
    for idx, (source_sentence, target_sentence) in \
            enumerate(tqdm(zip(source_sentences, target_sentences))):
        if len(source_sentence.strip().split()) > max_tokens or \
                len(target_sentence.strip().split()) > max_tokens:
            exclude_idx_list.append(idx)

    for idx in exclude_idx_list[::-1]:
        del source_sentences[idx]
        del target_sentences[idx]

    return source_sentences, target_sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_paths', type=str, nargs='+', required=True,
                        help='list of source input path')
    parser.add_argument('--target_paths', type=str, nargs='+', required=True,
                        help='list of target input path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='path to save directory')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='maximum tokens')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    for source, target in zip(args.source_paths, args.target_paths):
        source = Path(source)
        target = Path(target)
        assert source.exists()
        assert target.exists()

        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        source_output_path = save_dir / source.name
        target_output_path = save_dir / target.name

        with source.open() as sip:
            source_sentences = [sentence.strip() for sentence in sip.readlines()]
        with target.open() as tip:
            target_sentences = [sentence.strip() for sentence in tip.readlines()]

        source_sentences, target_sentences = limit_num_tokens_parallel(
            source_sentences,
            target_sentences,
            args.max_tokens,
        )

        with source_output_path.open('w') as sop:
            for sentence in source_sentences:
                sop.write(sentence + '\n')

        with target_output_path.open('w') as top:
            for sentence in target_sentences:
                top.write(sentence + '\n')
