import torch.nn as nn
import json
import argparse
from transformers import BertModel, BertConfig


# def setup_train_args():

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_config', default='Config/config.json', type=str, required=False,
#                         help='Choose_model_parameters')
#     parser.add_argument('--hidden_size', default=512, type=int, required=False)
#     parser.add_argument('--vocab_size', default=50000, type=int, required=False)
#     parser.add_argument('--check_dir', default='Config/check.txt', type=str, required=False)
#     parser.add_argument('--gpu', default=1, type=int, required=False)
#     parser.add_argument('--epochs', default=30, type=int, required=False)
#     parser.add_argument('--num_gradients', default=4, type=int, required=False)
#     parser.add_argument('--batch_size', default=16, type=int, required=False)
#     parser.add_argument('--lr', default=1e-5, type=int, required=False)
#     parser.add_argument('--load_dir', default='weights/english/no_kg/', type=str, required=False)
#     parser.add_argument('--validate_load_dir', default='../data/validate_data.pkl', type=str, required=False)
#     parser.add_argument('--train_load_dir', default='../data/train_data.pkl', type=str, required=False)
#     parser.add_argument('--log_dir', default='log/train.txt', type=str, required=False)
#     parser.add_argument('--val_epoch_interval', default=1, type=int, required=False)
#     return parser.parse_args()

class transformers_model(nn.Module):
    def __init__(self, config, hidden_size, vocab_size):
        super().__init__()
        # args = setup_train_args()
        self.config = config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        encoder_config = BertConfig.from_json_file(self.config)
        self.encoder = BertModel(encoder_config)

        decoder_config = BertConfig.from_json_file(self.config)
        decoder_config.is_decoder = True
        print(decoder_config)
        encoder_config.add_cross_attention = True
        decoder_config.add_cross_attention = True

        self.decoder = BertModel(decoder_config)

        self.linear = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        # encoder_hidden_states: [batch_size, max_length, hidden_size]
        #encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
        # out: [batch_size, max_length, hidden_size]
        out = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        out = out.last_hidden_state
        out = self.linear(out)
        return out


