import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW
from kg_model import transformers_model
from transformers import BertTokenizer


import nltk
nltk.download('wordnet')
import fire
from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams
from kg_dataloader import KgDataLoader

from torch_geometric.data import DataLoader as KGDataLoader



def bleu(predict, target, n):
    return sentence_bleu([target], predict, weights=tuple(1 / n for i in range(n)))


def nist(predict, target, n):
    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def cal_length(sentences):
    sen_length = [len(s.split()) for s in sentences]
    return np.mean(sen_length), np.var(sen_length)


def calculate_metrics(predict, reference):
    reference_len = len(reference)
    predict_len = len(predict)

    #-------------------bleu----------
    bleu_2 = bleu(predict, reference, 2)
    bleu_4 = bleu(predict, reference, 4)
    #-------------------nist----------
    nist_2 = nist(predict, reference, 2)
    nist_4 = nist(predict, reference, 4)
    #-------------------meteor----------
    predict = " ".join(predict)
    reference = " ".join(reference)
    meteor_scores = meteor_score([reference], predict)
    return bleu_2, bleu_4, nist_2, nist_4, meteor_scores


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
           #r_tags.append(tags[index])
   return r

# def sample_generate(

if __name__ == '__main__':
    # fire.Fire(sample_generate)
    top_k = 50
    temperature = 1.0
    decoder_path='weights/english/kg/29model.pth'#'decoder.pth'
    gpu_id=0
    # ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD VALIDATE DATA------------------
    test_data = torch.load("../data/test_data.pkl")

    # test_data = torch.load("../../../ijcnn/datasets/topical_chat_kb_2/test_freq/sentences.pkl")
    test_dataset = TensorDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

    kg_test_data = torch.load("processed/data_test.pt")

    kg_dataset_test = KgDataLoader('.')

    kg_loader_test = KGDataLoader(kg_dataset_test, batch_size=1, shuffle=False)

    # kg_model = Net(kg_dataset_train.num_node_features)

    # kg_model.to(device)

    #------------------------END LOAD VALIDATE DATA--------------

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',\
     never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[END]"))

    model = transformers_model(kg_dataset_test.num_node_features)

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


    checkpoint = torch.load(decoder_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

    #device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------



    #------------------------START GENERETE-------------------
    update_count = 0

    bleu_2scores = 0
    bleu_4scores = 0
    nist_2scores = 0
    nist_4scores = 0

    pred_file = open("pred_kg.txt", 'w')

    meteor_scores = 0
    sentences = []
    print('start generating....')
    batch_id = 0
    for batch, batch_kg in zip(test_dataloader, kg_loader_test):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]
            batch_kg = batch_kg.to(device)

            kg_input = batch_kg

            kg_hidden_states = model.kg_model(kg_input)

            encoder_input, decoder_input, mask_encoder_input, _ = batch

            past = model.encoder(encoder_input, mask_encoder_input)
            past = past.last_hidden_state
            # print('past',past.size())
            hidden_states = torch.cat((past.squeeze(0), kg_hidden_states),dim=0)

            prev_pred = decoder_input[:, :1]
            sentence = prev_pred

            # decoding loop
            for i in range(100):
                logits = model.decoder(sentence, encoder_hidden_states=hidden_states.unsqueeze(0))
                #print(logits)
                logits = logits.last_hidden_state
                logits = model.linear(logits)
                #logits = logits.last_hidden_state
                logits = logits[:, -1]
                logits = logits.squeeze(1) / temperature

                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence= torch.cat([sentence, prev_pred], dim=-1)
                if prev_pred[0][0] == 102:
                    break

           # print(sentence[0])
            predict = tokenizer.convert_ids_to_tokens(sentence[0].tolist())

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()

            reference = tokenizer.convert_ids_to_tokens(decoder_input[:decoder_input_num].tolist())

            reference = convert_to_original_length(reference)
            predict = convert_to_original_length(predict)
            if batch_id == 0 or batch_id == 1:
                print("##################################################\n")
                print('reference', reference)
                # print("\n\n")
                print('predict', predict)
                print("\n")
                
            batch_id += 1
            # if batch_id == 5000:
            #     break


            pred_file.write('-'*20 + f"example {update_count}" + '-'*20)
            pred_file.write(f"input: {' '.join(inputs)}")
            pred_file.write('\n')
            pred_file.write(f"output: {' '.join(reference)}")
            pred_file.write('\n')
            pred_file.write(f"predict: {' '.join(predict)}")
            pred_file.write('\n')
            

            pred_file.write("\n\n")
            temp_bleu_2, \
            temp_bleu_4, \
            temp_nist_2, \
            temp_nist_4, \
            temp_meteor_scores = calculate_metrics(predict[1:-1], reference[1:-1])

            bleu_2scores += temp_bleu_2
            bleu_4scores += temp_bleu_4
            nist_2scores += temp_nist_2
            nist_4scores += temp_nist_4

            meteor_scores += temp_meteor_scores
            sentences.append(" ".join(predict[1:-1]))
            update_count += 1

    entro, dist = cal_entropy(sentences)
    mean_len, var_len = cal_length(sentences)
    print(f'avg: {mean_len}, var: {var_len}')
    print(f'entro: {entro}')
    print(f'dist: {dist}')
    print(f'test bleu_2scores: {bleu_2scores / update_count}')
    print(f'test bleu_4scores: {bleu_4scores / update_count}')
    print(f'test nist_2scores: {nist_2scores / update_count}')
    print(f'test nist_4scores: {nist_4scores / update_count}')
    print(f'test meteor_scores: {meteor_scores / update_count}')
    pred_file.close()


