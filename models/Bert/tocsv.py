import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import argparse
import numpy as np
import pandas as pd
import nlgeval

nltk.download('wordnet')
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from model import transformers_model
from pytorch_pretrained_bert import BertTokenizer


def cal_length(sentences):
    sen_length = [len(s.split()) for s in sentences]
    return np.mean(sen_length), np.var(sen_length)


def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def convert_to_original_length(sentence):
    r = []
    r_tags = []

    for index, token in enumerate(sentence):
        if token.startswith("##"):
            if r:
                r[-1] = f"{r[-1]}{token[2:]}"
        else:
            r.append(token)
            # r_tags.append(tags[index])
    return r


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='Config/config.json', type=str, required=False,
                        help='Choose_model_parameters')
    parser.add_argument('--gpu', default=2, type=int, required=False)
    parser.add_argument('--top_k', default=50, type=int, required=False)
    parser.add_argument('--temp', default=1.0, type=float, required=False)
    parser.add_argument('--decoder_dir', default='weights/english/no_kg/covid/bestmodel.pth', type=str, required=False)
    parser.add_argument('--train_load_dir', default='../../preprocessed_data/data/c_data/train_data.pkl', type=str,
                        required=False)
    parser.add_argument('--test_load_dir', default='../../preprocessed_data/data/c_data/test_data.pkl', type=str,
                        required=False)
    parser.add_argument('--validate_load_dir', default='../../preprocessed_data/data/c_data/validate_data.pkl',
                        type=str,
                        required=False)


    # parser.add_argument('--pred_save_dir', default='Results/no_kg/med/pred.txt', type=str, required=False)
    parser.add_argument('--reference_save_dir', default='../../Raw/covid/reference.txt', type=str, required=False)
    # parser.add_argument('--metric_save_dir', default='Results/no_kg/med/scores.txt', type=str, required=False)
    parser.add_argument('--output_save_dir', default='../../Raw/covid/Full.csv', type=str, required=False)
    parser.add_argument('--output_trainsrc_save_dir', default='../../../../ACL2020-ConKADI/health_data/raw_files/train.src', type=str, required=False)
    parser.add_argument('--output_testsrc_save_dir', default='../../../../ACL2020-ConKADI/health_data/raw_files/test.src', type=str, required=False)
    parser.add_argument('--output_validatesrc_save_dir', default='../../../../ACL2020-ConKADI/health_data/raw_files/validate.src', type=str,
                        required=False)
    parser.add_argument('--output_traintgt_save_dir', default='../../../../ACL2020-ConKADI/health_data/raw_files/train.tgt', type=str, required=False)
    parser.add_argument('--output_testtgt_save_dir', default='../../../../ACL2020-ConKADI/health_data/raw_files/test.tgt', type=str, required=False)
    parser.add_argument('--output_validatetgt_save_dir', default='../../../../ACL2020-ConKADI/health_data/raw_files/validate.tgt', type=str,
                        required=False)
    parser.add_argument('--hidden_size', default=512, type=int, required=False)
    parser.add_argument('--vocab_size', default=50000, type=int, required=False)
    args = parser.parse_args()

    top_k = args.top_k
    temperature = args.temp
    decoder_path = args.decoder_dir
    train_path = args.train_load_dir
    test_path = args.test_load_dir
    validate_path = args.validate_load_dir

    device = torch.device(f"cuda:{args.gpu}")

    print('load the model....')
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', \
                                              never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[END]"))

    model = transformers_model(args.model_config, args.hidden_size, args.vocab_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, \
        lr=1e-5, \
        weight_decay=0.01,
    )

    checkpoint = torch.load(decoder_path, map_location='cuda:2')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

    # device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')

    train_data = torch.load(train_path)
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1)

    test_data = torch.load(test_path)
    test_dataset = TensorDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

    validate_data = torch.load(validate_path)
    validate_dataset = TensorDataset(*validate_data)
    validate_dataloader = DataLoader(dataset=validate_dataset, shuffle=False, batch_size=1)
    print('load success')

    # temp_train = {'src': [], 'trg': []}
    # temp_test = {'src': [], 'trg': []}
    # temp_validate = {'src': [], 'trg': []}
    # temp = {'src': [], 'trg': []}

    temp_train_src = {'src': []}
    temp_test_src = {'src': []}
    temp_validate_src = {'src': []}
    temp_train_tgt = {'trg': []}
    temp_test_tgt = {'trg': []}
    temp_validate_tgt = {'trg': []}
    temp = {'src': [], 'trg': []}

    ref_path = args.reference_save_dir

    print('start generating....')

    for batch_id, batch in enumerate(train_dataloader):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, _ = batch

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()
            reference = tokenizer.convert_ids_to_tokens(decoder_input[:decoder_input_num].tolist())

            inputs = convert_to_original_length(inputs)
            reference = convert_to_original_length(reference)

            if batch_id == 0 or batch_id == 1:
                print("##################################################\n")
                print('trg', reference)
                print("\n")

            temp['src'].append(' '.join(inputs[1:-1]))
            temp['trg'].append(' '.join(reference[1:-1]))

            temp_train_src['src'].append(' '.join(inputs[1:-1]))
            temp_train_tgt['trg'].append(' '.join(reference[1:-1]))

            # print(f"{' '.join(reference[1:-1])}", file=reference_file)

    for batch_id, batch in enumerate(test_dataloader):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, _ = batch

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()
            reference = tokenizer.convert_ids_to_tokens(decoder_input[:decoder_input_num].tolist())

            inputs = convert_to_original_length(inputs)
            reference = convert_to_original_length(reference)

            if batch_id == 0 or batch_id == 1:
                print("##################################################\n")
                print('trg', reference)
                print("\n")

            temp['src'].append(' '.join(inputs[1:-1]))
            temp['trg'].append(' '.join(reference[1:-1]))

            temp_test_src['src'].append(' '.join(inputs[1:-1]))
            temp_test_tgt['trg'].append(' '.join(reference[1:-1]))

            # print(f"{' '.join(reference[1:-1])}", file=reference_file)

    for batch_id, batch in enumerate(validate_dataloader):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, _ = batch

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()
            reference = tokenizer.convert_ids_to_tokens(decoder_input[:decoder_input_num].tolist())

            inputs = convert_to_original_length(inputs)
            reference = convert_to_original_length(reference)

            if batch_id == 0 or batch_id == 1:
                print("##################################################\n")
                print('trg', reference)
                print("\n")

            temp['src'].append(' '.join(inputs[1:-1]))
            temp['trg'].append(' '.join(reference[1:-1]))

            temp_validate_src['src'].append(' '.join(inputs[1:-1]))
            temp_validate_tgt['trg'].append(' '.join(reference[1:-1]))
            # print(f"{' '.join(reference[1:-1])}", file=reference_file)

    train_srcf = open(args.output_trainsrc_save_dir, 'w')
    for e in temp_train_src['src']:
        train_srcf.write(e + '\n')

    test_srcf = open(args.output_testsrc_save_dir, 'w')
    for e in temp_test_src['src']:
        test_srcf.write(e + '\n')

    validate_srcf = open(args.output_validatesrc_save_dir, 'w')
    for e in temp_validate_src['src']:
        validate_srcf.write(e + '\n')

    train_tgtf = open(args.output_traintgt_save_dir, 'w')
    for e in temp_train_tgt['trg']:
        train_tgtf.write(e + '\n')

    test_tgtf = open(args.output_testtgt_save_dir, 'w')
    for e in temp_test_tgt['trg']:
        test_tgtf.write(e + '\n')

    validate_tgtf = open(args.output_validatetgt_save_dir, 'w')
    for e in temp_validate_tgt['trg']:
        validate_tgtf.write(e + '\n')

    # train_save_path = args.output_trainsrc_save_dir
    # df = pd.DataFrame(temp_train_src)
    # df.to_csv(train_save_path, mode='w')
    #
    # test_save_path = args.output_testsrc_save_dir
    # df = pd.DataFrame(temp_test_src)
    # df.to_csv(test_save_path, mode='w')
    #
    # validate_save_path = args.output_validatesrc_save_dir
    # df = pd.DataFrame(temp_validate_src)
    # df.to_csv(validate_save_path, mode='w')
    #
    # train_save_path_1 = args.output_traintgt_save_dir
    # df = pd.DataFrame(temp_train_tgt)
    # df.to_csv(train_save_path_1, mode='w')
    #
    # test_save_path_1 = args.output_testtgt_save_dir
    # df = pd.DataFrame(temp_test_tgt)
    # df.to_csv(test_save_path_1, mode='w')
    #
    # validate_save_path_1 = args.output_validatetgt_save_dir
    # df = pd.DataFrame(temp_validate_tgt)
    # df.to_csv(validate_save_path_1, mode='w')
    #
    # full_path = args.output_save_dir
    # df = pd.DataFrame(temp)
    # df.to_csv(full_path, mode='w')
