import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from kg_model import transformers_model, Net
from kg_dataloader import KgDataLoader
import fire
import time
import os
from torch_geometric.data import DataLoader as KGDataLoader

# uses allennlp modules
from allennlp.nn import util

max_grad_norm = 1.0


if __name__ == '__main__':
    epochs=30
    num_gradients_accumulation=4
    batch_size=1
    gpu_id=0
    lr=1e-5
    load_dir='weights/english/kg/'
    save_every = 10
    device = torch.device(f"cuda:{gpu_id}")



    #------------------------LOAD TRAIN DATA------------------
    train_data = torch.load("../data/train_data.pkl")
    kg_train_data = torch.load("processed/data_train.pt")

    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    kg_dataset_train = KgDataLoader('.')
    kg_loader_train = KGDataLoader(kg_dataset_train, batch_size=batch_size, shuffle=False)

    val_data = torch.load("../data/validate_data.pkl")
    kg_val_data = torch.load("processed/data_validate.pt")

    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    kg_dataset_val = KgDataLoader('.')
    kg_loader_val = KGDataLoader(kg_dataset_val, batch_size=batch_size, shuffle=False)

    # print(kg_dataset_train.num_node_features)
    # kg_model = Net(kg_dataset_train.num_node_features)
    # kg_model.to(device)

    #------------------------END LOAD TRAIN DATA--------------

    #------------------------LOAD MODEL-----------------
    print('load the model....')

    model = transformers_model(kg_dataset_train.num_node_features)
    device = torch.device(f"cuda:0")
    model.to(device)
    print(model)

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,\
        lr=lr,\
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, \
        num_warmup_steps=num_train_optimization_steps // 10, \
        num_training_steps=num_train_optimization_steps
    )
    #------------------------END SET OPTIMIZER--------------


    #------------------------START TRAINING-------------------
    update_count = 0

   # PATH = "weights/english/3model.pth"
   # checkpoint = torch.load(PATH)
   # model.load_state_dict(checkpoint['model_state_dict'])
   # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   # epoch = checkpoint['epoch']
   # loss = checkpoint['loss']
   # model.eval()
    start = time.time()
    print('start training....') 
    for epoch in range(epochs):
        #------------------------training------------------------
        model.train()
        losses = 0
        times = 0
        for batch, batch_kg in zip(train_dataloader, kg_loader_train):
            # print(update_count)
            batch = [item.to(device) for item in batch]
            # print('batch',batch_kg.to(device))
            # for item in batch_kg:
            #     print('i',item)
            batch_kg = batch_kg.to(device)
            # batch_kg = [item.to(device) for item in batch_kg]
            kg_input = batch_kg
            # print(kg_input)

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, kg_input)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()

            times += 1
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'loss: {losses / times}')
        start = end

        #------------------------validate------------------------
        model.eval()

        perplexity = 0
        batch_count = 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            for batch, batch_kg in zip(val_dataloader, kg_loader_val):
                batch = [item.to(device) for item in batch]
                batch_kg = batch_kg.to(device)

                kg_input = batch_kg


                encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
                logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, kg_input)

                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder_input[:, 1:].contiguous()

                loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

                perplexity += np.exp(loss.item())

                batch_count += 1

        print(f'validate perplexity: {perplexity / batch_count}')

        direct_path = os.path.join(os.path.abspath('.'), load_dir)
        if not os.path.exists(direct_path):
            os.mkdir(direct_path)
        if (epoch + 1) % save_every == 0:
            print('saving model')
            torch.save( {'epoch' : epoch, 'model_state_dict' : model.state_dict(), 'optimizer_state_dict': optimizer.state_dict() , 'loss' : loss}  , os.path.join(direct_path, str(epoch) + "model.pth"))

    #------------------------END TRAINING-------------------


