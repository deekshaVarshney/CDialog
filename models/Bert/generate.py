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
from nlgeval import compute_metrics
from f1_score import F1_Score


# import fire
# from collections import defaultdict
#
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from nltk.translate.nist_score import sentence_nist
# from nltk.util import ngrams



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


# def sample_generate(



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='Config/config.json', type=str, required=False,
                        help='Choose_model_parameters')
    parser.add_argument('--gpu', default=3, type=int, required=False)
    parser.add_argument('--top_k', default=50, type=int, required=False)
    parser.add_argument('--temp', default=1.0, type=float, required=False)
    parser.add_argument('--decoder_dir', default='weights/english/no_kg/ext_data/bestmodel.pth', type=str, required=False)
    parser.add_argument('--test_load_dir', default='../../preprocessed_data/data_entity_random/test_data.pkl', type=str, required=False)
    parser.add_argument('--pred_save_dir', default='Results/no_kg/ext_data/pred.txt', type=str, required=False)
    parser.add_argument('--reference_save_dir', default='Results/no_kg/ext_data/reference.txt', type=str, required=False)
    parser.add_argument('--metric_save_dir', default='Results/no_kg/ext_data/scores.txt', type=str, required=False)
    parser.add_argument('--output_save_dir', default='Results/no_kg/ext_data/Out.csv', type=str, required=False)
    parser.add_argument('--hidden_size', default=512, type=int, required=False)
    parser.add_argument('--vocab_size', default=50000, type=int, required=False)

    args = parser.parse_args()

    # fire.Fire(sample_generate)
    top_k = args.top_k
    temperature = args.temp
    decoder_path = args.decoder_dir  # 'decoder.pth'
    gpu_id = args.gpu
    test_path = args.test_load_dir
    print(decoder_path)
    print(test_path)
    # ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    # ------------------------LOAD MODEL-----------------
    print('load the model....')
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', \
                                              never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[END]"))

    model = transformers_model(args.model_config, args.hidden_size, args.vocab_size)
    #model = torch.nn.DataParallel(model)

    # ----------------LOAD  OPTIMIZER-------------------
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

    checkpoint = torch.load(decoder_path, map_location='cuda:3')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

    # device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')
    # ------------------------END LOAD MODEL--------------

    # ------------------------LOAD VALIDATE DATA------------------
    test_data = torch.load(test_path)

    # test_data = torch.load("../../../ijcnn/datasets/topical_chat_kb_2/test_freq/sentences.pkl")
    test_dataset = TensorDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)
    # ------------------------END LOAD VALIDATE DATA--------------

    # ------------------------START GENERETE-------------------
    update_count = 0

    temp = {'input': [], 'reference': [], 'prediction': []}

    pred_path = args.pred_save_dir
    ref_path = args.reference_save_dir
    score_path = args.metric_save_dir

    pred_file = open(pred_path, 'w')
    reference_file = open(ref_path, 'w')
    # out_file = open("Results/no_kg/out.txt", 'w')
    score = open(score_path, 'w')

    # meteor_scores = 0
    sentences = []
    print('start generating....')
    for batch_id, batch in enumerate(test_dataloader):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, _ = batch

            past = model.encoder(encoder_input, mask_encoder_input)
            past = past.last_hidden_state

            prev_pred = decoder_input[:, :1]
            sentence = prev_pred

            # decoding loop
            for i in range(100):
                logits = model.decoder(sentence, encoder_hidden_states=past)
                # print(logits)
                logits = logits.last_hidden_state
                logits = model.linear(logits)
                # logits = logits.last_hidden_state
                logits = logits[:, -1]
                logits = logits.squeeze(1) / temperature

                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence = torch.cat([sentence, prev_pred], dim=-1)
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

            inputs = convert_to_original_length(inputs)
            reference = convert_to_original_length(reference)
            predict = convert_to_original_length(predict)

            if len(predict) == 2:
                predict.insert(1, '.')

            if batch_id == 0 or batch_id == 1:
                print("##################################################\n")
                print('reference', reference)
                # print("\n\n")
                print('predict', predict)
                print("\n")

            temp['input'].append(' '.join(inputs[1:-1]))
            temp['reference'].append(' '.join(reference[1:-1]))
            temp['prediction'].append(' '.join(predict[1:-1]))
            # out_file.write('-' * 20 + f"example {update_count}" + '-' * 20)
            # out_file.write(f"input: {' '.join(inputs)}")
            # out_file.write(f"output: {' '.join(reference)}")
            # out_file.write(f"predict: {' '.join(predict)}")
            # out_file.write("\n\n")

            print(f"{' '.join(reference[1:-1])}", file=reference_file)
            print(f"{' '.join(predict[1:-1])}", file=pred_file)

            sentences.append(" ".join(predict[1:-1]))
            update_count += 1

            # if batch_id == 10:
            #      break

    pred_file.close()
    reference_file.close()
    # out_file.close()

    out_path = args.output_save_dir

    df = pd.DataFrame(temp)
    df.to_csv(out_path, mode='w')

    #Computing the metric scores
    metrics_dict = compute_metrics(hypothesis=pred_path, references=[ref_path])
    f1_score = F1_Score(temp['reference'], temp['prediction'])

    score.write("\n\n")
    print(metrics_dict, file=score)
    score.write("\n")
    print("f1_score: ", f1_score)
    print("f1_score: ", file=score)
    print(f1_score, file=score)
    score.close()
