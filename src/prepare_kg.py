# File "/Data/aizan/.local/lib/python3.6/site-packages/quickumls/core.py"
#https://github.com/DATEXIS/UMLSParser follow this for creating UMLS-FILE
from nltk.corpus import stopwords
from quickumls import QuickUMLS
from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import codecs
from tokenizer import Tokenizer

import json
import numpy as np
import pandas as pd
import re
import networkx as nx
import pickle

MAX_ENCODER_SIZE = 400
MAX_DECODER_SIZE = 100

nltk_stopwords = stopwords.words('english')
tokenizer = Tokenizer('nltk')
tot = 0
maxi = 0
mini = 1e5

concept_dic = pickle.load(open('concept_dic_full.pkl','rb'))

print(len(list(concept_dic.keys())))
print(list(concept_dic.keys())[:10])


embeddings_index = dict()
f = open('glove.840B.300d.txt')

for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

def match(sent1, sent2):
    sent1 = sent1[8:].split()
    sent2 = sent2.split()
    common = set(sent1).intersection( set(sent2))
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


def get_edge_index(kg):
    source = [i[0] for i in kg] 
    target = [i[2] for i in kg]
    relations = [i[1] for i in kg]

    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
    H = nx.from_pandas_edgelist(kg_df, "source", "target", 
                              edge_attr=True, create_using=nx.Graph())

    edges = nx.to_scipy_sparse_matrix(H, format='coo')

    source  = edges.row.tolist()
    target = edges.col.tolist()
    return [source, target]
    
def get_feature_matrix(kg):
    source = [i[0] for i in kg]
    target = [i[2] for i in kg]

    count = 0
    nodes = set(source + target)
    emb_dim = 300
    embedding_matrix = np.zeros((len(nodes), emb_dim), dtype=np.float32)
    for i, word in enumerate(nodes):
        try:
          embedding_vector = embeddings_index[word]
        except:
          embedding_vector = np.zeros(emb_dim)
          count = count + 1

        embedding_matrix[i] = embedding_vector

    emb_weights = torch.from_numpy(embedding_matrix).cuda()

    return emb_weights

def return_kg(text):
    text = input_text_to_umls(text)
    words = set([re.sub(r'[^\w\s]','',word.lower()) for word in text.split() if re.sub(r'[^\w\s]','',word.lower()) not in nltk_stopwords and re.sub(r'[^\w\s]','',word.lower()) in concept_dic.keys()])
    # print('words',words)
    triples_list = []
    for word in words:
        # print('w',word)
        for triples in list(concept_dic[word]):
            # print(triples)

            if word in triples[1] and word != triples[1]:
                # print(triples)
                triples_list.append((word, triples[0], triples[1]))
    # print('triples_list',triples_list)
    return triples_list

def input_text_to_umls(input):
  text=''
  for i,j in enumerate(input):
    text +=j.split(':',1)[1]
  return text

def make_dataset(data, file_name='train_data.pth'):
    print(file_name)
    train_data = []
    kg_data = []
    count = 0
    print('couu',len(data))
    for d in data:
        print(count)
        d_len = len(d)
        
        for i in range(d_len // 2):
            #print('encoder_input: \n',d[:2 * i + 1])
            text = d[:2 * i + 1]
            # print(text)
            kg = return_kg(text)
            # print(kg)
            kg = kg[:10]
            try:
                edge_index_list = get_edge_index(kg)
                feature_matrix = get_feature_matrix(kg)
            except:
                kg = [('none','none','none')]
                edge_index_list = get_edge_index(kg)
                feature_matrix = get_feature_matrix(kg)

            # print('ee',edge_index_list)
            # print('fm',feature_matrix)

            # edge_index_list = torch.LongTensor(edge_index_list)
            # feature_matrix = torch.LongTensor(feature_matrix)

            kg_data.append((edge_index_list, feature_matrix))

        # if count == 10:
        #     break
        count += 1

    torch.save(kg_data, 'Transformer/raw/kg_'+ file_name)

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

print(len(data[0]))
print(len(data[1]))
print(len(data[2]))

print(f'Process the train dataset')
make_dataset(data[0], 'train_data.pkl')

print(f'Process the validate dataset')
make_dataset(data[1], 'validate_data.pkl')

print(f'Process the test dataset')
make_dataset(data[2], 'test_data.pkl')


