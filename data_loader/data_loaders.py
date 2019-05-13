from typing import Callable, Dict, Sequence, Union, List
from pathlib import Path

import torch
from torch.utils.data import Dataset

from base import BaseDataLoader
from preprocessor import TextPreprocessor
from .collate_functions import seq2seq_collate_fn


class Seq2SeqDataset(Dataset):
    '''
    Dataset for seq2seq
    '''
    __slot__ = [
        'src_list',
        'tgt_list',
        'tgt_langs',
        'tgt_styles'
        'text_preprocessor',
    ]

    def __init__(
            self,
            src_list: List[List[str]],
            tgt_list: List[List[str]],
            tgt_langs: List[int],
            tgt_styles: List[int],
            text_preprocessor: TextPreprocessor,
    ):
        '''
        create seq2seq dataset.

        :param src_list: nested list of source text
        :param tgt_list: nested list of target text
        :param tgt_langs: target languages to be translated from source sentences.
        :parma tgt_styles: target styles to be translated from source sentences.
        :param text_preprocessor: text preprocessor
        '''
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.text_preprocessor = text_preprocessor
        self.tgt_langs = tgt_langs
        self.tgt_styles = tgt_styles

        assert len(src_list) == len(tgt_list)
        assert len(src_list) == len(tgt_langs)
        assert len(src_list) == len(tgt_styles)
        assert text_preprocessor.num_languages == len(set(tgt_langs))
        assert text_preprocessor.num_styles == len(set(tgt_styles))

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        src_tokens = self.src_list[idx].split()
        src_indices = self.text_preprocessor.tokens2indice(src_tokens, sos=False, eos=False)
        src_indices = torch.Tensor(src_indices)

        tgt_tokens = self.tgt_list[idx].split()
        tgt_indices = self.text_preprocessor.tokens2indice(tgt_tokens, sos=True, eos=True)
        tgt_indices = torch.Tensor(tgt_indices)


        tgt_lang = self.text_preprocessor.lang2index(self.tgt_langs[idx])
        tgt_lang = torch.LongTensor([tgt_lang])
        tgt_style = self.text_preprocessor.style2index(self.tgt_styles[idx])
        tgt_style = torch.LongTensor([tgt_style])

        return src_indices, tgt_indices, tgt_lang, tgt_style

    @classmethod
    def create(
            cls,
            source_paths: List[Union[str, Path]],
            target_paths: List[Union[str, Path]],
            target_langs: List[str],
            target_styles: List[str],
            text_preprocessor: TextPreprocessor,
    ) -> 'Seq2SeqDataset':
        '''
        create seq2seq dataset from text paths

        :param source_paths: list of paths to source sentences
        :param target_paths: list of paths to target sentences
        :param target_langs: target langauges from source sentences to be translated.
        :param target_styles: target styles from source sentences to be translated.
        :param text_preprocessor: text preprocessor

        NOTE
        ----
        The format of target_langs and target_styles are like below tihs.
        [<target_style1>, <target_style2>, <target_style_3>]
        [<target_lang1>, <target_lang2>, <target_lang3>]
        '''
        assert len(source_paths) == len(target_paths)
        assert len(source_paths) == len(target_langs)
        assert len(source_paths) == len(target_styles)

        for idx in range(len(source_paths)):
            if isinstance(source_paths[idx], str):
                source_paths[idx] = Path(source_paths[idx])
            if isinstance(target_paths[idx], str):
                target_paths[idx] = Path(target_paths[idx])
            assert source_paths[idx].exists()
            assert target_paths[idx].exists()

        source_text_list = []
        target_text_list = []
        languages = []
        styles = []

        for source_path, target_path, lang, style in zip(source_paths, target_paths, target_langs, target_styles):
            with source_path.open() as f:
                source_sub_text_list = [text.strip().lower() for text in f.readlines()]
                source_text_list += source_sub_text_list

            with target_path.open() as f:
                target_sub_text_list = [text.strip().lower() for text in f.readlines()]
                target_text_list += target_sub_text_list

            # append target language and style for num-sentence times.
            languages += [lang]*len(source_sub_text_list)
            styles += [style]*len(source_sub_text_list)

            assert len(source_text_list) == len(target_text_list)
            assert len(source_text_list) == len(languages)
            assert len(source_text_list) == len(styles)

        return cls(
            source_text_list,
            target_text_list,
            languages,
            styles,
            text_preprocessor,
        )


class Seq2seqDataLoader(BaseDataLoader):
    '''
    Seq2Seq data loader using BaseDataLoader
    '''
    def __init__(
            self,
            src_paths: Sequence[Union[str, Path]],
            tgt_paths: Sequence[Union[str, Path]],
            tgt_languages: Sequence[str],
            tgt_styles: Sequence[str],
            text_preprocessor_path: Union[str, Path],
            batch_size: int=1,
            shuffle: bool=True,
            validation_split: float=0.0,
            num_workers: int=1,
            collate_fn: Callable=seq2seq_collate_fn,
    ):
        '''
        DataLoader for seq2seq data

        :param src_path: list of paths to source sentences
        :param tgt_path: list of paths to target sentences
        :param tgt_languages: list of languages to be translated from source sentences.
        :param tgt_styles: list of styles to be translated from source sentences.
        :param text_preprocessor_path: path to text preprocessor
        :param batch_size: batch size
        :param shuffle: shuffle data
        :param validation_split: split dataset for validation
        :param num_workers: the number of workers
        '''
        assert len(src_paths) == len(tgt_paths)
        assert len(src_paths) == len(tgt_languages)
        assert len(src_paths) == len(tgt_styles)

        for idx in range(len(src_paths)):
            if isinstance(src_paths[idx], str):
                src_paths[idx] = Path(src_paths[idx])
            if isinstance(tgt_paths[idx], str):
                tgt_paths[idx] = Path(tgt_paths[idx])
            assert src_paths[idx].exists(), src_paths[idx].as_posix()
            assert tgt_paths[idx].exists(), tgt_paths[idx].as_posix()

        if isinstance(text_preprocessor_path, str):
            text_preprocessor_path = Path(text_preprocessor_path)
        assert text_preprocessor_path.exists()

        self.src_paths = src_paths
        self.tgt_paths = tgt_paths
        self.text_preprocessor_path = text_preprocessor_path
        self.text_preprocessor = TextPreprocessor.load(text_preprocessor_path)

        self.dataset = Seq2SeqDataset.create(
            src_paths,
            tgt_paths,
            tgt_languages,
            tgt_styles,
            self.text_preprocessor,
        )

        super(Seq2seqDataLoader, self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=collate_fn,
        )
