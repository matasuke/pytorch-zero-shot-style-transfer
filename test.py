import os
import json
import argparse
import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.model as module_arch
import model.translator as module_translate
from config_parser import ConfigParser, CustomArgs


def main(config: ConfigParser):
    # setup data_loader instances
    data_loader = config.initialize('test_data_loader', module_data)

    # build model architecture
    model = config.initialize('arch', module_arch)
    print(model)

    # load state dict
    print(f'Loading checkpoint: {config.resume}')
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model, device_ids=config.device, dim=1)
        model.load_state_dict(state_dict)
        model = model.module
    else:
        model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # create translator by wrapping model
    translator = config.initialize(
        'translator',
        module_translate,
        model,
        data_loader.text_preprocessor
    )

    # prepare output file
    out_f = (config.test_dir / config['translator']['output']).open('w')

    with torch.no_grad():
        for batch_idx, (src, tgt, tgt_lang, tgt_style, lengths, indices) in enumerate(tqdm(data_loader)):
            src, tgt = src.to(device), tgt.to(device)
            tgt_lang, tgt_style = tgt_lang.to(device), tgt_style.to(device)

            pred_batch = translator.translate(src, tgt_lang, tgt_style, lengths, indices)

            for b in range(len(pred_batch)):
                out_f.write(' '.join(pred_batch[b][0]) + '\n')

    out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str, required=True,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, required=True,
                        help='indices of GPUs to enable (default: all)')

    # optional arguments
    options = [
        CustomArgs(['-dt', '--decode_type'], type=str, target=('translator', 'args', 'decoder_type'), choices=['greedy', 'beam']),
        CustomArgs(['-max_len', '--max_length'], type=int, target=('translator', 'args', 'max_length')),
        CustomArgs(['-beam', '--beam_width'], type=int, target=('translator', 'args', 'beam_width')),
        CustomArgs(['-n_best', '--n_best'], type=int, target=('translator', 'args', 'n_best')),
        CustomArgs(['-out', '--output'], type=str, target=('translator', 'output')),
    ]

    config = ConfigParser.parse_args(parser, options)
    main(config)
