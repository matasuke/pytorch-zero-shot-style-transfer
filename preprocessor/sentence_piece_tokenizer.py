from argparse import ArgumentParser
from pathlib import Path
import shutil
import subprocess
import pickle
import hashlib
from typing import Sequence, List, Union

import tqdm
import sentencepiece as spm


DEFAULT_VOCAB_SIZE = 8000
# 0.9995 for languages with rich character set like Japanese or Chinese.
# and 1.0 for other languages with small character set.
DEFAULT_COVERAGE = 0.9995
MODEL_TYPES = ['unigram', 'bpe', 'char', 'word']
DEFAULT_MODEL_TYPE = 'unigram'


class SentencePieceTokenizer:
    '''
    preprocessor based on sentencepiece.
    see https://github.com/google/sentencepiece for more detail.
    '''
    __slot__ = ['_model_path', '_vocab_path', '_sp', '_vocab_size']

    PAD_ID = 0  # padding
    SOS_ID = 1  # start of sentence
    EOS_ID = 2  # end of sentence
    UNK_ID = 3  # unknown word
    START_ID = 4  # start of word id

    PAD_SYMBOL = '<PAD>'
    SOS_SYMBOL = '<SOS>'
    EOS_SYMBOL = '<EOS>'
    UNK_SYMBOL = '<UNK>'

    IDS = [PAD_ID, SOS_ID, EOS_ID, UNK_ID]
    SYMBOLS = [PAD_SYMBOL, SOS_SYMBOL, EOS_SYMBOL, UNK_SYMBOL]

    MAX_SAMPLE_SIZE = 100000
    MODEL_PREFIX = 'spm'
    MODEL_FILE_NAME = MODEL_PREFIX + '.model'
    VOCAB_FILE_NAME = MODEL_PREFIX + '.vocab'
    TMP_DIR_PREFIX = Path('.tmp/')
    TMP_FILE_NAME = 'data.txt'

    def __init__(
            self,
            model_path: Union[str, Path],
            vocab_path: Union[str, Path],
    ) -> None:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()
        if isinstance(vocab_path, Path):
            vocab_path = vocab_path.as_posix()

        self._model_path = model_path
        self._vocab_path = vocab_path
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._vocab_size = len(self._sp)

    @property
    def vocab_size(self) -> int:
        'get vocabulary size'
        return self._vocab_size

    @classmethod
    def create(  # type: ignore
            cls,
            text_list: Sequence[str],
            max_vocab_size: int=DEFAULT_VOCAB_SIZE,
            model_type: str=DEFAULT_MODEL_TYPE,
            coverage: float=DEFAULT_COVERAGE,
            sampling: bool=False,
    ) -> 'SentencePieceTokenizer':
        '''
        create text features using sentencepiece.

        :param text_list: List of sentences for training.
        :param max_vocab_size: max vocaburaly size.
        :param model_type: model tyoe for sentencepiece.
        :param coverage: amout of characters covered by the model.
            good defaults are 0.9995 for languages with rich character set like
            Japanese or Chinese and 1.0 for other languages with small character set.
        :param sampling: ramdomly samples MAX_SAMPLE_SIZE of sentences for memory efficiency.
        :return: SentencePieceTokenizer
        '''
        assert len(text_list) == 0 or isinstance(text_list[0], str)
        assert model_type in MODEL_TYPES

        hash_seed = str(text_list) + str(max_vocab_size) + model_type
        setting_hash = hashlib.md5(hash_seed.encode('utf-8')).hexdigest()
        tmp_dir = cls.TMP_DIR_PREFIX / setting_hash
        model_path = tmp_dir / cls.MODEL_FILE_NAME
        vocab_path = tmp_dir / cls.VOCAB_FILE_NAME

        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)
        if not model_path.exists() and not vocab_path.exists():
            cls._create_model(
                text_list,
                tmp_dir,
                max_vocab_size,
                model_type,
                coverage,
                sampling,
            )
        else:
            raise Exception('file exists in ' + tmp_dir.as_posix())

        return cls(model_path, vocab_path)

    @classmethod
    def _create_model(
            cls,
            text_list: Sequence[str],
            tmp_dir: Union[str, Path],
            vocab_size: int,
            model_type: str=DEFAULT_MODEL_TYPE,
            coverage: float=DEFAULT_COVERAGE,
            sampling: bool=False,
    ) -> None:
        '''
        create sentence piece preprocessor.
        '''
        if isinstance(tmp_dir, str):
            tmp_dir = Path(tmp_dir)
        tmp_model_prefix = tmp_dir / cls.MODEL_PREFIX
        tmp_file_path = tmp_dir / cls.TMP_FILE_NAME

        input_sentence_size = len(text_list)
        if sampling:
            input_sentence_size = cls.MAX_SAMPLE_SIZE

        _text_list = '\n'.join(text_list)
        with tmp_file_path.open('w') as f:
            f.write(_text_list)

        command_list = {
            'input': tmp_file_path.as_posix(),
            'model_prefix': tmp_model_prefix.as_posix(),
            'vocab_size': vocab_size,
            'pad_id': cls.PAD_ID,
            'bos_id': cls.SOS_ID,
            'eos_id': cls.EOS_ID,
            'unk_id': cls.UNK_ID,
            'pad_piece': cls.PAD_SYMBOL,
            'bos_piece': cls.SOS_SYMBOL,
            'eos_piece': cls.EOS_SYMBOL,
            'unk_piece': cls.UNK_SYMBOL,
            'unk_surface': cls.UNK_SYMBOL,
            'character_coverage': coverage,
            'model_type': model_type,
            'input_sentence_size': input_sentence_size,
            'add_dummy_prefix': 'false'
        }
        # parse command_list for SentencePieceTrainer
        command = ''
        for key, value in command_list.items():
            command += '--' + str(key) + '=' + str(value) + ' '

        # train spm
        spm.SentencePieceTrainer.train(command)

    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> 'SentencePieceTokenizer':
        '''
        Load saved SentencePieceTokenizer

        :param load_dir: path to saved model dir.
        '''
        if isinstance(load_dir, str):
            load_dir = Path(load_dir)
        assert load_dir.exists()

        model_path = load_dir / cls.MODEL_FILE_NAME
        vocab_path = load_dir / cls.VOCAB_FILE_NAME

        if not model_path.exists() or not vocab_path.exists():
            raise Exception('file does not exists it' + load_dir.as_posix())

        return cls(model_path, vocab_path)

    def save(self, save_dir: Union[str, Path]) -> None:
        '''
        save trained SentencePieceTokenizer

        :param save_path: path to save model
        '''
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        for (from_path, to_path) in ([
                (self._model_path, save_dir / self.MODEL_FILE_NAME),
                (self._vocab_path, save_dir / self.VOCAB_FILE_NAME),
        ]):
            if from_path != to_path:
                shutil.copyfile(from_path, to_path)

    def tokenize(self, text_list: Sequence[str]) -> List[str]:
        '''
        tokenize list of sentences.

        :param text_list: list of sentences.
        :return: list of tokenized sentences.
        '''
        tokenized_list = []
        for text in text_list:
            tokenized = self._sp.encode_as_pieces(text)
            tokenized_list.append(tokenized)

        return tokenized_list


if __name__ == '__main__':
    parser = ArgumentParser('create tokenizer for seq2seq model')
    parser.add_argument('--data_paths', type=str, nargs='+', required=True,
                        help='list of path to text dataset')
    parser.add_argument('--tokenizer_save_dir', type=str, required=True,
                        help='path to save model and tokenized')
    parser.add_argument('--data_save_dir', type=str, required=True,
                        help='path to save model and tokenized')
    parser.add_argument('--vocab_size', type=int, default=DEFAULT_VOCAB_SIZE,
                        help='vocaburaly size')
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE,
                        choices=MODEL_TYPES,
                        help='model type to train SentencePiece tokenizer')
    parser.add_argument('--coverage', type=str, default=DEFAULT_COVERAGE,
                        help='the amount of coverage')
    parser.add_argument('--sampling', action='store_true',
                        help='maximum sample size for memory efficiency')
    args = parser.parse_args()

    data_paths = []
    for data_path in args.data_paths:
        data_path = Path(data_path)
        data_paths.append(data_path)
        assert data_path.exists(), data_path.as_posix()

    tokenizer_save_dir = Path(args.tokenizer_save_dir)
    if not tokenizer_save_dir.exists():
        tokenizer_save_dir.mkdir(parents=True)
    data_save_dir = Path(args.data_save_dir)
    if not data_save_dir.exists():
        data_save_dir.mkdir(parents=True)

    print('Loading text...')
    sentences = []
    split_points = []  # set split points of data lines for tokenize and save sentences.
    for data_path in data_paths:
        with data_path.open() as f:
            sentences += [sentence.strip() for sentence in f.readlines()]
            split_points.append(len(sentences))
    print(f'text size: {len(sentences)}')

    print('Lowering sentences...')
    for idx, sentence in enumerate(sentences):
        sentences[idx] = sentence.lower()

    print('Creating SentencePieceTokenizer...')
    tokenizer = SentencePieceTokenizer.create(
        text_list=sentences,
        max_vocab_size=args.vocab_size,
        model_type=args.model_type,
        coverage=args.coverage,
        sampling=args.sampling,
    )
    tokenizer.save(tokenizer_save_dir)
    tokenized = tokenizer.tokenize(sentences)

    # save tokenized file
    prev_split_point = 0
    for data_path, split_point in zip(data_paths, split_points):
        tokenized_save_path = data_save_dir / data_path.name
        sentences = tokenized[prev_split_point:split_point]
        prev_split_point = split_point

        with tokenized_save_path.open('w') as f:
            for line in sentences:
                f.write(' '.join(line) + '\n')
