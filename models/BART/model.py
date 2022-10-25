import torch.nn as nn
import argparse
from transformers import BartModel, BartConfig, BartForCausalLM
from transformers import BartTokenizer, BartForCausalLM

class transformers_model(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_config = BartConfig(
            vocab_size=50265,
            encoder_attention_heads=8,
            decoder_attention_heads=8
        )
        self.encoder = BartModel(encoder_config)


        # decoder_config = BartConfig(
        #     vocab_size=50265,
        #     encoder_attention_heads=8,
        #     decoder_attention_heads=8
        # )
        # decoder_config.is_decoder = True
        # print(decoder_config)
        # encoder_config.add_cross_attention=True
        # decoder_config.add_cross_attention=True
        # decoder_config.output_hidden_states=True
        # self.decoder = BartForCausalLM.from_pretrained('facebook/bart-large', config=decoder_config)
        self.linear = nn.Linear(1024, 50265, bias=False)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        # encoder_hidden_states: [batch_size, max_length, hidden_size]
        #encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        #encoder_hidden_states = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = self.encoder(input_ids, mask_encoder_input,decoder_input_ids= output_ids,decoder_attention_mask= mask_decoder_input)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
        # out: [batch_size, max_length, hidden_size] torch.Size([2, 400, 1024])
        # out = self.decoder(input_ids=output_ids, attention_mask=mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        # out = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        # out = out.hidden_states
        out = self.linear(encoder_hidden_states)
        return out


