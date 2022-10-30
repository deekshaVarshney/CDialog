from transformers import AutoTokenizer
import torch
import os
import codecs
import json
import csv
import glob
import random
# tokenizer = AutoTokenizer.from_pretrained("BioBert/weights/biobert_weight/biobert_v1.1_pubmed")

tokenizer = AutoTokenizer.from_pretrained("models/BioBert/weights/biobert_weight/dmis_biobert_large_case")

MAX_ENCODER_SIZE = 400
MAX_DECODER_SIZE = 100

tot = 0
maxi = 0
mini = 1e5


def match(sent1, sent2):
    sent1 = sent1[8:].split()
    sent2 = sent2.split()
    # print('ss1',sent1)
    # print('ss2',sent2)

    common = set(sent1).intersection(set(sent2))
    # print('c',common)
    # print(len(common)/(len(set(sent1))))
    #
    if len(common) / len(set(sent1)) > 0.90:
        # print('True')
        return True
    else:
        return False


def clean_dataset(dataset_file, json_file):

    rows = []
    f_in = open(dataset_file, "r", encoding='cp1252')
    # with open(dataset_file, "r", encoding='utf-8', errors='ignore') as file:
    csvreader = csv.reader(f_in)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)


    f_json = open(json_file, "w", encoding='utf-8')
    # row[1] = conv_id
    # row[2] = speaker
    # row[3] = turn_num
    # row[5] = utterance
    # row[6] = entity

    total = 0
    last_turn = 0
    last_dialog = {}
    last_list = []
    Dialog_list = []
    id = "1"
    sen = ""
    print(len(rows))
    for lst in rows:
        if lst[1] != id:
            last_dialog["Turn"] = int(int(last_turn)/2)
            last_dialog["Id"] = int(id)
            last_dialog["Dialogue"] = last_list[:]
            Dialog_list.append(last_dialog.copy())
            last_list.clear()
        id = lst[1]
        last_turn = lst[3]
        sen = lst[2] .strip() + ": " + lst[5].strip()
        sen = sen.strip()
        last_list.append(sen)
        #print(len(Dialog_list))
    # print(len(Dialog_list))
    # print(Dialog_list[1])
    last_dialog["Turn"] = int(int(last_turn)/2)
    last_dialog["Id"] = int(id)
    last_dialog["Dialogue"] = last_list[:]
    Dialog_list.append(last_dialog.copy())


    # print(Dialog_list[0])
    # print(last_list)

    print(len(Dialog_list))

    # print("Total Cases: ", json_file.split('.')[0].split('/')[-1], id)
    json.dump(Dialog_list, f_json, indent=4)
    f_in.close()
    f_json.close()
    return id


def seq2token_ids(source_seqs, target_seq):
    encoder_input = []
    for source_seq in source_seqs:
        # print('sss',source_seq[8:])
        encoder_input += tokenizer.tokenize(source_seq[8:]) + ["[SEP]"]

    decoder_input = ["[CLS]"] + tokenizer.tokenize(target_seq[7:])  # ?? xx:
    # print(encoder_input)
    # print(decoder_input)

    if len(encoder_input) > MAX_ENCODER_SIZE - 1:
        if "[SEP]" in encoder_input[-MAX_ENCODER_SIZE:-1]:
            idx = encoder_input[:-1].index("[SEP]", -(MAX_ENCODER_SIZE - 1))
            encoder_input = encoder_input[idx + 1:]

    encoder_input = ["[CLS]"] + encoder_input[-(MAX_ENCODER_SIZE - 1):]
    decoder_input = decoder_input[:MAX_DECODER_SIZE - 1] + ["[SEP]"]
    enc_len = len(encoder_input)
    dec_len = len(decoder_input)

    # conver to ids
    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)

    # mask
    mask_encoder_input = [1] * len(encoder_input)
    mask_decoder_input = [1] * len(decoder_input)

    global tot, maxi, mini

    tot = tot + len(encoder_input) + len(decoder_input)
    maxi = max(maxi, len(encoder_input))
    maxi = max(maxi, len(decoder_input))

    mini = min(mini, len(encoder_input))
    mini = min(mini, len(decoder_input))

    # padding
    encoder_input += [0] * (MAX_ENCODER_SIZE - len(encoder_input))
    decoder_input += [0] * (MAX_DECODER_SIZE - len(decoder_input))
    mask_encoder_input += [0] * (MAX_ENCODER_SIZE - len(mask_encoder_input))
    mask_decoder_input += [0] * (MAX_DECODER_SIZE - len(mask_decoder_input))

    # turn into tensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)

    mask_encoder_input = torch.LongTensor(mask_encoder_input)
    mask_decoder_input = torch.LongTensor(mask_decoder_input)

    return encoder_input, decoder_input, mask_encoder_input, mask_decoder_input


def make_dataset(data, file_name='train_data.pth'):
    train_data = []
    count = 0
    for d in data:
        # print(count)
        d_len = len(d)
        for i in range(d_len // 2):
            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[:2 * i + 1],
                                                                                                 d[2 * i + 1])
            train_data.append((encoder_input,
                               decoder_input,
                               mask_encoder_input,
                               mask_decoder_input))
        # break
        count += 1

    encoder_input, \
    decoder_input, \
    mask_encoder_input, \
    mask_decoder_input = zip(*train_data)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    mask_encoder_input = torch.stack(mask_encoder_input)
    mask_decoder_input = torch.stack(mask_decoder_input)

    train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input]

    torch.save(train_data, file_name)


def get_splited_data_by_file(dataset_file):
    datasets = [[], [], []]

    with open(dataset_file, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)


    total_id_num = len(data)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    datasets[0] = [d['Dialogue'] for d in data[:validate_idx]]
    datasets[1] = [d['Dialogue'] for d in data[validate_idx:test_idx]]
    datasets[2] = [d['Dialogue'] for d in data[test_idx:]]

    # print(datasets)
    return datasets


path1 = 'Data/Ext_data/'
path3 = 'Data/json_files_biobert/'

total = 0

tot_data = [[], [], []]


json_file = 'Data_entity/json_files/merged/merged_file.json'

temp = get_splited_data_by_file(json_file)
tot_data[0].extend(temp[0])
tot_data[1].extend(temp[1])
tot_data[2].extend(temp[2])

# data = get_splited_data_by_file(json_file)

data = tot_data

# print(data[0][100])


print(len(data[0]))
print(len(data[1]))
print(len(data[2]))

print(f'Process the train dataset')
make_dataset(data[0], 'preprocessed_data/data_biobert_random/train_data.pkl')

print(f'Process the validate dataset')
make_dataset(data[1], 'preprocessed_data/data_biobert_random/validate_data.pkl')

print(f'Process the test dataset')
make_dataset(data[2], 'preprocessed_data/data_biobert_random/test_data.pkl')

print("#############")
print(tot)
print(maxi)
print(mini)
