import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
from model import transformers_model
# from kg_dataloader import KgDataLoader
import fire
import time
import os

# uses allennlp modules
from allennlp.nn import util

max_grad_norm = 1.0

# def train_model(

if __name__ == '__main__':
    # print(train_model)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='Config/config.json', type=str, required=False,
                    help='Choose_model_parameters')
    parser.add_argument('--gpu', default=3, type=int, required=False)
    parser.add_argument('--epochs', default=30, type=int, required=False)
    parser.add_argument('--num_gradients', default=4, type=int, required=False)
    parser.add_argument('--batch_size', default=32, type=int, required=False)
    parser.add_argument('--lr', default=1e-5, type=int, required=False)
    parser.add_argument('--load_dir', default='weights/english/no_kg/ext_data/', type=str, required=False)
    parser.add_argument('--validate_load_dir', default='../../preprocessed_data/data_entity_random/validate_data.pkl', type=str, required=False)
    parser.add_argument('--train_load_dir', default='../../preprocessed_data/data_entity_random/train_data.pkl', type=str, required=False)
    parser.add_argument('--log_dir', default='log/train.txt', type=str, required=False)
    parser.add_argument('--val_epoch_interval', default=1, type=int, required=False)
    parser.add_argument('--last_epoch_path', default="/home1/deeksha/CDialog/src/models/Bert/weights/english/no_kg/covid_fine/", type=str, required=False)
    parser.add_argument('--hidden_size', default=512, type=int, required=False)
    parser.add_argument('--vocab_size', default=50000, type=int, required=False)
    parser.add_argument('--finetune', default= 'false', type=str, required=False)
    args = parser.parse_args()
    #30522
    epochs = args.epochs
    num_gradients_accumulation = args.num_gradients
    batch_size = args.batch_size
    gpu_id = args.gpu
    lr = args.lr
    load_dir = args.load_dir
    validate_load = args.validate_load_dir
    train_load = args.train_load_dir
    log_directory = args.log_dir
    valid_epoch = args.val_epoch_interval

    print(train_load)
    print(validate_load)
    save_every = 10
    # ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")
    # ------------------------LOAD MODEL-----------------
    print('load the model....')
    # device = torch.device(f"cuda")
    model = transformers_model(args.model_config, args.hidden_size, args.vocab_size)
    # device = torch.device(f"cuda:{2}")
    #model = torch.nn.DataParallel(model)
    model.to(device)

    print('load success')
    # ------------------------END LOAD MODEL--------------

    # ------------------------LOAD TRAIN DATA------------------
    train_data = torch.load(train_load)
    #
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)

    val_data = torch.load(validate_load)

    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    # ------------------------END LOAD TRAIN DATA--------------

    # ------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, \
        lr=lr, \
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, \
        num_warmup_steps=num_train_optimization_steps // 10, \
        num_training_steps=num_train_optimization_steps
    )
    # ------------------------END SET OPTIMIZER--------------

    # ------------------------START TRAINING-------------------
    update_count = 0

    finetune_check = args.finetune
    PATH = args.last_epoch_path
    if finetune_check == 'false':
        if not os.listdir(PATH):
            print("Training the model from starting")
        else:
            files_int = list()
            for i in os.listdir(PATH):
                if 'best' not in i:
                    epoch = int(i.split('model.')[0])
                    if epoch == 29:
                        files_int.clear()
                        break
                    else:
                        files_int.append(epoch)

            if len(files_int) == 0:
                print("No valid file found from which training should resume")
            else:
                max_value = max(files_int)
                for i in os.listdir(PATH):
                    if 'best' not in i:
                        epoch = int(i.split('model.')[0])
                    if epoch > max_value:
                        pass
                    elif epoch < max_value:
                        pass
                    else:
                        final_file = i
                    print(f'Resuming training from epoch {max_value}')
                    checkpoint = torch.load(PATH + final_file)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    loss = checkpoint['loss']
                    valid_epoch = epoch
                    model.eval()


    if finetune_check == 'true' or finetune_check == 'True':
        print("Initiating finetuning")
        checkpoint = torch.load(PATH + 'bestmodel.pth', map_location='cuda:3')
        print(PATH + 'bestmodel.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        model.eval()

    f = open(log_directory, "w")
    f.close()

    start = time.time()
    best_valid_perplexity = float('inf')
    best_valid_epoch = 0
    best_valid_loss = 0
    print('start training....')
    for epoch in range(epochs):
        # ------------------------training------------------------
        f = open(log_directory, "a")
        model.train()
        losses = 0
        times = 0
        for batch in train_dataloader:
            # print(update_count)
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

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
        print('-' * 20 + f'epoch {epoch}' + '-' * 20)
        print('-' * 20 + f'epoch {epoch}' + '-' * 20, file=f)
        print(f'time: {(end - start)}')
        print(f'time: {(end - start)}', file=f)
        print(f'loss: {losses / times}')
        print(f'loss: {losses / times}', file=f)
        start = end

        # ------------------------validate------------------------
        # Calculating the perplexity when no of epoch == valid_epoch
        if (epoch + 1) % valid_epoch == 0:
            model.eval()

            perplexity = 0
            batch_count = 0
            print('start calculate the perplexity....')
            print('start calculate the perplexity....', file=f)
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

            cur_valid_perplexity = perplexity / batch_count
            cur_valid_loss = loss

            print(f'validate perplexity: {perplexity / batch_count}')
            print(f'validate perplexity: {perplexity / batch_count}', file=f)

            if cur_valid_perplexity < best_valid_perplexity:
                best_valid_perplexity = cur_valid_perplexity
                best_valid_epoch = epoch
                best_valid_loss = cur_valid_loss
                direct_path = os.path.join(os.path.abspath('.'), load_dir)
                if not os.path.exists(direct_path):
                    os.mkdir(direct_path)

                # save_range = range(epochs-5, epochs)
                # if epoch in save_range:
                print(f'saving best model having epoch: {best_valid_epoch}')
                print(f'saving best model having epoch: {best_valid_epoch}', file=f)
                torch.save(
                    {'epoch': best_valid_epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': best_valid_loss}, os.path.join(direct_path, "bestmodel.pth"))

        direct_path = os.path.join(os.path.abspath('.'), load_dir)
        if not os.path.exists(direct_path):
            os.mkdir(direct_path)
        # if (epoch + 1) % save_every == 0:
        if epoch == 29:
            print('saving model')
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss}, os.path.join(direct_path, str(epoch) + "model.pth"))

    f.close()

    # ------------------------END TRAINING-------------------
