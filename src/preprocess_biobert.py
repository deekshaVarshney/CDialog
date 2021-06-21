from transformers import AutoTokenizer
import torch
import os
import codecs
import json

tokenizer = AutoTokenizer.from_pretrained("BioBert/weights/biobert_weight/biobert_v1.1_pubmed")

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

    common = set(sent1).intersection( set(sent2))
    # print('c',common)
    # print(len(common)/(len(set(sent1))))
# 
    if len(common)/len(set(sent1)) > 0.90:
        # print('True')
        return True
    else:
        return False

def clean_dataset(dataset_file, json_file):
    f_in = open(dataset_file, "r")
    f_json = open(json_file, "w", encoding='utf-8')
    
    total = 0
    last_part = ""
    last_turn = 0
    last_dialog = {}
    last_list = []
    last_user = ""
    
    Dialog_list = []
    
    check_list = []
    
    while True:
        line = f_in.readline()
        # print('llll',line)
        if not line:
            break
        if line[:11] == "Description":
            # print('lll',line[:11])

            # print('jj')
            last_part = "description"
            last_turn = 0
            last_dialog = {}
            last_list = []
            last_user = ""
            last_utterance = ""
            while True:
                line = f_in.readline()
                if (not line) or (line in ["\n", "\n\r"]):
                    break
                # print('linee',line[:])
                # print('dddd',line) 

                # if line[:11] == 'Description':
                # print('ssssssss')
                last_user = "Patient:"

                sen = line.rstrip()
                if sen == "":
                    continue
                if sen[-1] not in '.。？，,?!！~～':
                    sen += '。'
                if sen in check_list:
                    last_utterance = ""
                else:
                    last_utterance = last_user + sen
                    check_list.append(sen)
                break

        elif line[:8] == "Dialogue":
            # print(last_part)
            if last_part == "description" and len(last_utterance) > 0:
                last_part = "dialogue"
                # last_user = "病人："
                last_user = "Patient:"

                last_turn = 1
                # print(line)
                while True:
                    line = f_in.readline()
                    # print('lll',line)
                    if (not line) or (line in ["\n", "\n\r"]):
                        last_user = ""
                        last_list.append(last_utterance)

                        if  int(last_turn / 2) > 0:
                            temp = int(last_turn / 2)
                            last_dialog["Turn"] = temp
                            total += 1
                            last_dialog["Id"] = total
                            last_dialog["Dialogue"] = last_list[: temp * 2]
                            Dialog_list.append(last_dialog)

                        break

                    # print('line',line[:8])
                    
                    if line[:8] == "Patient:" or line[:7] == "Doctor:":

                        user = line[:8]
                        line = f_in.readline()
                        sen = line.rstrip()
                        if sen == "":
                            continue

                        if sen[-1] not in '.。？，,?!！~～':
                            sen += '。'
                            
                        if user == last_user:
                            if match(last_utterance,sen):
                                last_utterance = last_utterance
                            else:
                                last_utterance = last_utterance + sen
                        else:
                            last_user = user
                            last_list.append(last_utterance)
                            last_turn += 1
                            last_utterance = user + sen

    print ("Total Cases: ", json_file.split('.')[0].split('/')[-1], total)
    json.dump(Dialog_list, f_json, ensure_ascii = False, indent = 4)
    f_in.close()
    f_json.close()
    return total


def seq2token_ids(source_seqs, target_seq):
    # 可以尝试对source_seq进行切分
    encoder_input = []
    for source_seq in source_seqs:
        # 去掉 xx：
        # print('sss',source_seq[8:])
        encoder_input += tokenizer.tokenize(source_seq[8:]) + ["[SEP]"]

    decoder_input = ["[CLS]"] + tokenizer.tokenize(target_seq[7:])  # 去掉 xx：
    # print(encoder_input)
    # print(decoder_input)

    # 设置不得超过 MAX_ENCODER_SIZE 大小
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
        print(count)
        d_len = len(d)
        for i in range(d_len // 2):
            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[:2 * i + 1], d[2 * i + 1])
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

path1 = 'MedDialogCorpus/HCM/'
path2 = 'MedDialogCorpus/Icliniq/'

path3 = 'MedDialogCorpus/json_files/'
total = 0

tot_data = [[], [], []]


for root,dirnames,filenames in os.walk(path1):
    for filename in filenames:
        dataset_file = os.path.join(os.path.abspath('../src/'), os.path.join(path1,filename))
        new_filename = filename.split('.txt')[0] + '.json'
        json_file = os.path.join(os.path.abspath('../src/'), os.path.join(path3,new_filename))
        nos = clean_dataset(dataset_file, json_file)

for root,dirnames,filenames in os.walk(path2):
    for filename in filenames:
        dataset_file = os.path.join(os.path.abspath('../src/'), os.path.join(path2,filename))
        new_filename = filename.split('.txt')[0] + '.json'
        json_file = os.path.join(os.path.abspath('../src/'), os.path.join(path3,new_filename))
        nos = clean_dataset(dataset_file, json_file)


for root,dirnames,filenames in os.walk(path3):
    for filename in filenames:
        json_file = os.path.join(os.path.abspath('../src/'), os.path.join(path3,filename))
        temp = get_splited_data_by_file(json_file)
        tot_data[0].extend(temp[0])
        tot_data[1].extend(temp[1])
        tot_data[2].extend(temp[2])


# print('Total_data_crawl',total)


#data = get_splited_data_by_file(json_file)

data = tot_data

print(data[0][100])


print(len(data[0]))
print(len(data[1]))
print(len(data[2]))

print(f'Process the train dataset')
make_dataset(data[0], 'data_biobert/train_data.pkl')

print(f'Process the validate dataset')
make_dataset(data[1], 'data_biobert/validate_data.pkl')

print(f'Process the test dataset')
make_dataset(data[2], 'data_biobert/test_data.pkl')

print("#############")
print(tot)
print(maxi)
print(mini)

