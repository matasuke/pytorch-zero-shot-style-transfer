from typing import List, Callable, Union, Tuple

import torch.nn as nn
import torch

from .model import GlobalAttention
from .model import Model
from .beam import Beam
from preprocessor.text_preprocessor import TextPreprocessor


TYPE_GREEDY = 'greedy'
TYPE_BEAM = 'beam'
DECODER_TYPE = [TYPE_GREEDY, TYPE_BEAM]


class Translator:
    '''
    wrapper class for GreedySearchDecoder and BeamSearchDecoder
    '''
    def __init__(
            self,
            model: Model,
            text_preprocessor: TextPreprocessor,
            decoder_type: str,
            max_length: int=50,
            beam_width: int=5,
            n_best: int=1,
    ):
        if decoder_type == TYPE_GREEDY:
            self.translator = GreedySearch(
                model,
                text_preprocessor,
                max_length,
            )
        elif decoder_type == TYPE_BEAM:
            self.translator = BeamSearch(
                model,
                text_preprocessor,
                max_length,
                beam_width,
                n_best,
            )
        else:
            msg = f'Unknown decoder type: {decoder_type}'
            raise ValueError(msg)

    def translate(
            self,
            src_batch: torch.Tensor,
            tgt_lang: torch.Tensor,
            tgt_style: torch.Tensor,
            lengths: List[int],
            indices: List[int],
    ) -> Tuple[List, List, List]:
        '''
        translate sentences basedon choosen decoder.

        :src_batch: source batch size of [max_len, batch_size]
        :tgt_lang: target langauges size of [batch_size, ]
        :tgt_styles: target styles size of [batch_size, ]
        :lengths: length of each source sentence
        :indices: indices to be used for sorting
        '''
        pred_batch = self.translator.translate(
            src_batch,
            tgt_lang,
            tgt_style,
            lengths,
            indices,
        )

        return pred_batch


class GreedySearch(nn.Module):
    '''
    Greedy search decoder for neural language generation.
    '''
    def __init__(
            self,
            model: Model,
            text_preprocessor: TextPreprocessor,
            max_length: int=50,
    ):
        super(GreedySearch, self).__init__()
        self.text_preprocessor = text_preprocessor
        self.max_length = max_length

        if torch.cuda.is_available():
            self.tt = torch.cuda
            self.model = model.cuda()
        else:
            self.tt = torch
            self.model = model.cpu()

        self.model.eval()

    def translate_batch(
            self,
            src_batch: torch.Tensor,
            tgt_lang: torch.Tensor,
            tgt_style: torch.Tensor,
            lengths: List[int],
    ) -> Tuple[List, List]:
        '''
        forward input through greedy search decoder

        :param src_batch: source batch size of [max_len, batch_size, hidden_dim]
        :param tgt_lang: target languages to be translated size of [batch_size,]
        :param tgt_style: target styles to be translated size of [batch_size,]
        :param lengths: list of length of source batch
        '''
        batch_size = src_batch.size(1)

        # (1) run the encoder on the src
        enc_hidden, context = self.model.encoder(src_batch, tgt_lang, tgt_style, lengths)

        dec_output = self.model.make_init_decoder_output(context)
        enc_hidden = (self.model._fix_enc_hidden(enc_hidden[0]),
                      self.model._fix_enc_hidden(enc_hidden[1]))

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        pad_mask = src_batch.data.eq(TextPreprocessor.PAD_ID).t()

        def apply_context_mask(mask: GlobalAttention):
            if isinstance(mask, GlobalAttention):
                mask.applyMask(pad_mask)

        dec_input = self.tt.LongTensor([[TextPreprocessor.SOS_ID for _ in range(batch_size)]])
        output_tokens = self.tt.LongTensor([[TextPreprocessor.SOS_ID for _ in range(batch_size)]])

        # (2) run the decoder to generate sentences using greedy search
        for _ in range(self.max_length):
            self.model.decoder.apply(apply_context_mask)
            dec_output, enc_hidden, _ = self.model.decoder(
                dec_input,
                enc_hidden,
                context,
                dec_output,
            )
            # [batch_size, hidden_dim] > [batch_size, num_words]
            out = self.model.generator.forward(dec_output.squeeze(0))
            _, dec_input = torch.max(out, dim=1)

            # prepare current token to be next decoder input
            dec_input = dec_input.unsqueeze(0)
            output_tokens = torch.cat([output_tokens, dec_input], dim=0)

        output_tokens = output_tokens[1:].t().tolist()

        return output_tokens

    def translate(
            self,
            src_batch: torch.Tensor,
            tgt_lang: torch.Tensor,
            tgt_style: torch.Tensor,
            lengths: List[int],
            indices: List[int],
    ) -> Tuple[List, List, List]:
        #  (1) translate
        pred = self.translate_batch(src_batch, tgt_lang, tgt_style, lengths)
        pred = list(zip(*sorted(zip(pred, indices), key=lambda x: x[-1])))[0]

        #  (2) convert indexes to words
        pred_batch = []
        for batch_idx in range(src_batch.size(1)):
            tokens = self.text_preprocessor.indice2tokens(pred[batch_idx], stop_eos=True)
            pred_batch.append([tokens])

        return pred_batch


class BeamSearch(nn.Module):
    def __init__(
            self,
            model: Model,
            text_preprocessor: TextPreprocessor,
            max_length: int=50,
            beam_width: int=5,
            n_best: int=1,
    ):
        super(BeamSearch, self).__init__()
        self.beam_width = beam_width
        self.n_best = n_best
        self.max_length = max_length
        self.text_preprocessor = text_preprocessor

        if torch.cuda.is_available():
            self.tt = torch.cuda
            self.model = model.cuda()
        else:
            self.tt = torch
            self.model = model.cpu()

        self.model.eval()

    def translateBatch(
            self,
            src_batch: torch.Tensor,
            tgt_lang: torch.Tensor,
            tgt_style: torch.Tensor,
            lengths: List[int],
    ):
        batch_size = src_batch.size(1)
        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(src_batch, tgt_lang, tgt_style, lengths)

        rnnSize = context.size(2)

        encStates = (self.model._fix_enc_hidden(encStates[0]),
                     self.model._fix_enc_hidden(encStates[1]))

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = src_batch.data.eq(TextPreprocessor.PAD_ID).t()

        def applyContextMask(m):
            if isinstance(m, GlobalAttention):
                m.applyMask(padMask)

        #  (2) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = context.data.repeat(1, self.beam_width, 1)
        decStates = (encStates[0].data.repeat(1, self.beam_width, 1),
                     encStates[1].data.repeat(1, self.beam_width, 1))

        beam = [Beam(self.beam_width) for k in range(batch_size)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = src_batch.data.eq(TextPreprocessor.PAD_ID).t().repeat(self.beam_width, 1)
        batchIdx = list(range(batch_size))
        remainingSents = batch_size

        for i in range(self.max_length):
            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input. [1, batch_size * beam_width]
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)
            decOut, decStates, attn = self.model.decoder(input, decStates, context, decOut)
            decOut = decOut.squeeze(0)  # decOut: [batch_size * beam_width, hidden_dim]
            out = self.model.generator.forward(decOut)  # out: [batch_size * beam_width, numWords]
            # [batch_size, beam_width, numWords]
            wordLk = out.view(self.beam_width, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(self.beam_width, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, self.beam_width, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)

            padMask = padMask.unsqueeze(2).view(self.beam_width, -1, padMask.size(1))
            padMask = padMask.index_select(1, activeIdx)
            padMask = padMask.view(self.beam_width*activeIdx.size(0), -1)

            remainingSents = len(active)

        #  (3) package everything up

        all_hyp = []
        n_best = self.n_best
        for b in range(batch_size):
            scores, ks = beam[b].sortBest()

            hyps = []
            for k in ks[:n_best]:
                hyp, _ = beam[b].getHyp(k)
                hyps.append(torch.stack(hyp).tolist())
            all_hyp += [hyps]

        return all_hyp

    def translate(
            self,
            src_batch: torch.Tensor,
            tgt_lang: torch.Tensor,
            tgt_style: torch.Tensor,
            lengths: List[int],
            indices: List[int],
    ) -> Tuple[List, List, List]:
        #  (2) translate
        pred = self.translateBatch(src_batch, tgt_lang, tgt_style, lengths)
        pred = list(zip(*sorted(zip(pred, indices), key=lambda x: x[-1])))[0]

        #  (3) convert indexes to words
        pred_batch = []
        for batch_idx in range(src_batch.size(1)):
            pred_batch.append(
                [self.text_preprocessor.indice2tokens(pred[batch_idx][n], stop_eos=True)
                 for n in range(self.n_best)]
            )

        return pred_batch
