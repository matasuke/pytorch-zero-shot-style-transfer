'''
delete sentences which have more than specified number of tokens.
'''
import argparse
from typing import List, Tuple
from pathlib import Path
import random
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--source_paths', type=str, nargs='+', required=True,
                    help='list of source input path')
parser.add_argument('--target_paths', type=str, nargs='+', required=True,
                    help='list of target input path')
parser.add_argument('--save_dir', type=str, required=True,
                    help='path to save directory')
parser.add_argument('--sample_size', type=int, default=10000,
                    help='maximum tokens')
args = parser.parse_args()

save_dir = Path(args.save_dir)
if not save_dir.exists():
    save_dir.mkdir(parents=True)

for source, target in zip(args.source_paths, args.target_paths):
    source = Path(source)
    target = Path(target)
    assert source.exists(), source.as_posix()
    assert target.exists(), target.as_posix()

    source_output_path = save_dir / source.name
    target_output_path = save_dir / target.name

    sip = source.open()
    tip = target.open()

    sentence_pairs = []
    for src_sen, tgt_sen in tqdm(zip(sip.readlines(), tip.readlines())):
        sentence_pairs.append([src_sen.strip(), tgt_sen.strip()])

    if len(sentence_pairs) > args.sample_size:
        sentence_pairs = random.sample(sentence_pairs, args.sample_size)

    source_sentences = []
    target_sentences = []
    for src_sen, tgt_sen in sentence_pairs:
        source_sentences.append(src_sen)
        target_sentences.append(tgt_sen)

    assert len(source_sentences) == len(target_sentences)
    with source_output_path.open('w') as sop:
        for sentence in source_sentences:
            sop.write(sentence + '\n')

    with target_output_path.open('w') as top:
        for sentence in target_sentences:
            top.write(sentence + '\n')
