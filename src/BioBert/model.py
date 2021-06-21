import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class transformers_model(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_config = AutoConfig.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed/config.json")
        
        self.encoder = AutoModel.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed",config=encoder_config)

        decoder_config = AutoConfig.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed/config.json")
        
        self.decoder = AutoModel.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed",config=decoder_config)
        decoder_config.is_decoder = True
        encoder_config.add_cross_attention=True
        decoder_config.add_cross_attention=True
        self.linear = nn.Linear(768, 28996, bias=False)

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
