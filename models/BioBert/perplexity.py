import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from model import transformers_model
import argparse
import fire
import time
import os

# uses allennlp modules
from allennlp.nn import util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=7, type=int, required=False)
parser.add_argument('--batch_size', default=1, type=int, required=False)
parser.add_argument('--load_dir', default='weights/english/no_kg/ext_data/bestmodel.pth', type=str, required=False)
parser.add_argument('--validate_load_dir', default='../../preprocessed_data/data_biobert_random/validate_data.pkl', type=str, required=False)
parser.add_argument('--test_load_dir', default='../../preprocessed_data/data_biobert_random/test_data.pkl', type=str, required=False)
parser.add_argument('--train_load_dir', default='../../preprocessed_data/data_biobert_random/train_data.pkl', type=str, required=False)
parser.add_argument('--save_dir', default='Results/no_kg/ext_data/ppl.txt', type=str, required=False)
parser.add_argument('--hidden_size', default=768, type=int, required=False)
parser.add_argument('--vocab_size', default=28996, type=int, required=False)

args = parser.parse_args()
def calculate(
    # batch_size=1,
    # gpu_id=0,
    decoder_path='decoder_model'
    ):
    batch_size = args.batch_size
    gpu_id = args.gpu
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")


    #------------------------LOAD MODEL-----------------
    print('load the model....')

    model = transformers_model()
    # model = transformers_model(args.hidden_size, args.vocab_size)


    #----------------LOAD  OPTIMIZER-------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,\
        lr=1e-5,\
        weight_decay=0.01,
    )


    PATH = args.load_dir
    checkpoint = torch.load(PATH, map_location='cuda:6')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

  #  model.load_state_dict(torch.load(decoder_path))
 #   device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
    train_data = torch.load(args.train_load_dir)
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    val_data = torch.load(args.validate_load_dir)
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    test_data = torch.load(args.test_load_dir)
    test_dataset = TensorDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
    #------------------------END LOAD TRAIN DATA--------------

    print(PATH)
    print(args.train_load_dir)
    print(args.validate_load_dir)
    print(args.test_load_dir)

    ppl_file = open(args.save_dir, 'w')

    #------------------------START TRAINING-------------------

    print('start training cal...')
    #------------------------training------------------------
    perplexity = 0
    batch_count = 0
    with torch.no_grad():
        for batch in train_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

            perplexity += np.exp(loss.item())

            batch_count += 1

    print(f'train perplexity: {perplexity / batch_count}')
    print(f'train perplexity: {perplexity / batch_count}', file=ppl_file)

    #------------------------validate------------------------

    perplexity = 0
    batch_count = 0
    print('start calculate the perplexity....')

    with torch.no_grad():
        for batch in val_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

            perplexity += np.exp(loss.item())

            batch_count += 1

    print(f'validate perplexity: {perplexity / batch_count}')
    print(f'validate perplexity: {perplexity / batch_count}', file=ppl_file)

    perplexity = 0
    batch_count = 0
    print('start calculate the test perplexity....')

    with torch.no_grad():
        for batch in test_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

            perplexity += np.exp(loss.item())

            batch_count += 1

    print(f'test perplexity: {perplexity / batch_count}')
    print(f'test perplexity: {perplexity / batch_count}', file=ppl_file)
    ppl_file.close()
    #------------------------END cal-------------------


if __name__ == '__main__':
    fire.Fire(calculate)

