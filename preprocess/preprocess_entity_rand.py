from pytorch_pretrained_bert import BertTokenizer
import torch
import csv
import os
import codecs
import json
import glob
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_size', default=400, type=int, required=False)
parser.add_argument('--decoder_size', default=100, type=int, required=False)
parser.add_argument('--datapath', default='Data_entity/Ext_data/', type=str, required=False)
# parser.add_argument('--Icliniq_datapath', default='MedDialogCorpus/Icliniq/', type=str, required=False)
parser.add_argument('--json_datapath', default='Data_entity/json_files/', type=str, required=False)
parser.add_argument('--json_entity', default='Data_entity/entity_json_files/', type=str, required=False)
parser.add_argument('--save', default='preprocessed_data/data_entity_random', type=str, required=False)
parser.add_argument('--merged_datapath', default='Data_entity/json_files/merged/merged_file.json', type=str, required=False)
parser.add_argument('--entity_merged_datapath', default='Data_entity/entity_json_files/merged/merged_entity_file.json', type=str, required=False)

args = parser.parse_args()

# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', \
                                          never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[END]"))

MAX_ENCODER_SIZE = args.encoder_size
MAX_DECODER_SIZE = args.decoder_size

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


def clean_entity(dataset_file, json_file):
    rows = []
    f_in = open(dataset_file, "r", encoding='utf-8')
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
    Entity_list = []
    id = "1"
    sen = ""
    print(len(rows))
    for lst in rows:
        if lst[1] != id:
            last_dialog["Turn"] = int(int(last_turn) / 2)
            last_dialog["Id"] = int(id)
            last_dialog["Dialogue"] = last_list[:]
            Entity_list.append(last_dialog.copy())
            last_list.clear()
        id = lst[1]
        last_turn = lst[3]
        sen = lst[6].strip() + " " + lst[7].strip()
        sen = sen.strip()
        sen = sen + " " + lst[8].strip() + " " + lst[9].strip()
        sen = sen.strip()
        sen = sen + " " + lst[10].strip() + " " + lst[11].strip()
        sen = sen.strip()
        sen = sen + " " + lst[12].strip()
        sen = sen.strip()
        last_list.append(sen)
        # print(len(Dialog_list))
    # print(len(Dialog_list))
    # print(Dialog_list[1])
    last_dialog["Turn"] = int(int(last_turn) / 2)
    last_dialog["Id"] = int(id)
    last_dialog["Dialogue"] = last_list[:]
    Entity_list.append(last_dialog.copy())

    # print(Dialog_list[0])
    # print(last_list)

    print(len(Entity_list))

    # print("Total Cases: ", json_file.split('.')[0].split('/')[-1], id)
    json.dump(Entity_list, f_json, indent=4)
    f_in.close()
    f_json.close()
    return id


def clean_dataset(dataset_file, json_file):
    rows = []
    f_in = open(dataset_file, "r", encoding='utf-8')
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
            last_dialog["Turn"] = int(int(last_turn) / 2)
            last_dialog["Id"] = int(id)
            last_dialog["Dialogue"] = last_list[:]
            Dialog_list.append(last_dialog.copy())
            last_list.clear()
        id = lst[1]
        last_turn = lst[3]
        sen = lst[2].strip() + ": " + lst[5].strip()
        sen = sen.strip()
        last_list.append(sen)
        # print(len(Dialog_list))
    # print(len(Dialog_list))
    # print(Dialog_list[1])
    last_dialog["Turn"] = int(int(last_turn) / 2)
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
    # ?????source_seq????
    encoder_input = []
    for source_seq in source_seqs:
        # ?? xx:
        # print('sss',source_seq[8:])
        encoder_input += tokenizer.tokenize(source_seq[8:]) + ["[SEP]"]

    decoder_input = ["[CLS]"] + tokenizer.tokenize(target_seq[7:])  # ?? xx:
    # print(encoder_input)
    # print(decoder_input)

    # ?????? MAX_ENCODER_SIZE ??
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


def make_dataset(data, entity, file_name='train_data.pth'):
    train_data = []
    count = 0

    for d, e in zip(data, entity):
        # print(count)

        d_len = len(d)
        e_len = len(e)

        for i in range(d_len // 2):
            # print('src', d[:2 * i + 1])
            # print('trg', d[2 * i + 1])

            # src = d[:2*i+1]
            lst1 = d[:2 * i + 1]
            lst2 = e[:2 * i + 1]
            src = []
            # print(len(lst1), " and ", len(lst2))
            # print(lst1, " and ", lst2)
            for j in range(len(lst1)):
                # sen = lst1[j].strip + " " + lst2[j].strip
                # sen = sen.strip
                if lst2[j] != '':
                    src.append(lst1[j] + " " + lst2[j])
                else:
                    src.append(lst1[j])

            # print("src: ", src)
            # print(d[:2 * i + 1])
            # print(d[2 * i + 1])
            # encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[:2 * i + 1],
            #                                                                                      d[2 * i + 1])
            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(src,
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

    # for d in data[:]:

    # lst = []
    # dialogue_len = 0
    # for x in d['Dialogue']:
    #     lst = x.split()
    #     dialogue_len += 1
    #     if len(lst) < 4:
    #         if dialogue_len == 2:
    #             data.remove(d)
    #             break
    # else:
    #     d['Dialogue'] = d['Dialogue'][:dialogue_len-2]

    total_id_num = len(data)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)
    print('total_id_num:', total_id_num)
    print('validate_idx:', validate_idx)
    print('test_idx:', test_idx)
    datasets[0] = [d['Dialogue'] for d in data[:validate_idx]]
    datasets[1] = [d['Dialogue'] for d in data[validate_idx:test_idx]]
    datasets[2] = [d['Dialogue'] for d in data[test_idx:]]
    # print(datasets)
    return datasets


path1 = args.datapath
path2 = args.json_entity
path3 = args.json_datapath

total = 0

for root, dirnames, filenames in os.walk(path1):
    for filename in filenames:
        dataset_file = os.path.join(os.path.abspath('../Ext_CDialog/'), os.path.join(path1, filename))
        print(dataset_file)
        new_filename = filename.split('.csv')[0] + '.json'
        json_file = os.path.join(os.path.abspath('../Ext_CDialog/'), os.path.join(path3, new_filename))
        nos = clean_dataset(dataset_file, json_file)

for root, dirnames, filenames in os.walk(path1):
    for filename in filenames:
        dataset_file = os.path.join(os.path.abspath('../Ext_CDialog/'), os.path.join(path1, filename))
        print(dataset_file)
        new_filename = filename.split('.csv')[0] + '.json'
        json_file = os.path.join(os.path.abspath('../Ext_CDialog/'), os.path.join(path2, new_filename))
        nos = clean_entity(dataset_file, json_file)

# for root, dirnames, filenames in os.walk(path2):
#     for filename in filenames:
#         dataset_file = os.path.join(os.path.abspath('../src/'), os.path.join(path2, filename))
#         new_filename = filename.split('.txt')[0] + '.json'
#         json_file = os.path.join(os.path.abspath('../src/'), os.path.join(path3, new_filename))
#         nos = clean_dataset(dataset_file, json_file)


data1 = []
for filename in glob.glob("Data_entity/json_files/*.json"):
    with open(filename, "r") as infile:
        data1.extend(json.load(infile))

entity1 = []
for filename in glob.glob("Data_entity/entity_json_files/*.json"):
    with open(filename, "r") as infile:
        entity1.extend(json.load(infile))


d_e = list(zip(data1, entity1))
random.shuffle(d_e)
data1, entity1 = zip(*d_e)


json_file1 = args.merged_datapath
json_file2 = args.entity_merged_datapath


with open(json_file1, "w") as outfile:
    json.dump(data1, outfile)

with open(json_file2, "w") as outfile:
    json.dump(entity1, outfile)

tot_data = [[], [], []]
temp = get_splited_data_by_file(json_file1)
tot_data[0].extend(temp[0])
tot_data[1].extend(temp[1])
tot_data[2].extend(temp[2])

# for root, dirnames, filenames in os.walk(path3):
#     for filename in filenames:
#         json_file = os.path.join(os.path.abspath('../Ext_CDialog/'), os.path.join(path3, filename))
#         temp = get_splited_data_by_file(json_file)
#         tot_data[0].extend(temp[0])
#         tot_data[1].extend(temp[1])
#         tot_data[2].extend(temp[2])
data = tot_data

tot_entity = [[], [], []]

temp2 = get_splited_data_by_file(json_file2)
tot_entity[0].extend(temp2[0])
tot_entity[1].extend(temp2[1])
tot_entity[2].extend(temp2[2])
# for root, dirnames, filenames in os.walk(path3):
#     for filename in filenames:
#         json_file = os.path.join(os.path.abspath('../Ext_CDialog/'), os.path.join(path2, filename))
#         temp = get_splited_data_by_file(json_file)
#         tot_entity[0].extend(temp[0])
#         tot_entity[1].extend(temp[1])
#         tot_entity[2].extend(temp[2])

entity = tot_entity

# print(data[0][100])
# print(data[2])

print(len(data[0]))
print(len(data[1]))
print(len(data[2]))

print("entity len: ")
print(len(entity[0]))
print(len(entity[1]))
print(len(entity[2]))

# count = 0
# for d, e in zip(data[0], entity[0]):
#     # print(count)
#     if count > 5:
#         break
#     count += 1
#     d_len = len(d)
#     e_len = len(e)
#     for i in range(d_len // 2):
#         # print('src', d[:2 * i + 1])
#         # print('trg', d[2 * i + 1])
#
#         lst1 = d[:2 * i + 1]
#         lst2 = e[:2 * i + 1]
#         src = []
#         sen = ""
#         print(len(lst1), " and ", len(lst2))
#         print(lst1, " and ", lst2)
#         for j in range(len(lst1)):
#             # sen = lst1[j].strip + " " + lst2[j].strip
#             # sen = sen.strip
#             if lst2[j] != '':
#                 src.append(lst1[j] + " " + lst2[j])
#             else:
#                 src.append(lst1[j])
#
#         print("src: ", src)
#         print(d[:2 * i + 1])
#         print(d[2 * i + 1])

print(f'Process the train dataset')
make_dataset(data[0],entity[0], args.save + '/train_data.pkl')

print(f'Process the validate dataset')
make_dataset(data[1],entity[1], args.save + '/validate_data.pkl')

print(f'Process the test dataset')
make_dataset(data[2],entity[2], args.save + '/test_data.pkl')

print("#############")
print(tot)
print(maxi)
print(mini)
