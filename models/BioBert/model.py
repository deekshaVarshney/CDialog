# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModel, AutoConfig
# import argparse

# class transformers_model(nn.Module):
#     def __init__(self, hidden_size, vocab_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#         encoder_config = AutoConfig.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed/config.json")
        
#         self.encoder = AutoModel.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed",config=encoder_config)

#         decoder_config = AutoConfig.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed/config.json")
        
#         self.decoder = AutoModel.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed",config=decoder_config)
#         decoder_config.is_decoder = True
#         encoder_config.add_cross_attention=True
#         decoder_config.add_cross_attention=True
#         self.linear = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

#     def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
#         # encoder_hidden_states: [batch_size, max_length, hidden_size]
#         #encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
#         encoder_hidden_states = self.encoder(input_ids, mask_encoder_input)
#         encoder_hidden_states = encoder_hidden_states.last_hidden_state
#         # out: [batch_size, max_length, hidden_size]
#         out = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
#         out = out.last_hidden_state
#         out = self.linear(out)
#         return out

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertModel, BertConfig
#from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig


class transformers_model(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder_config = BertConfig(
        #     num_hidden_layers=6,
        #     vocab_size=50000,
        #     hidden_size=512,
        #     num_attention_heads=8
        # )
        #------ BioBert Config and Model loading------------
        #encoder_config = AutoConfig.from_pretrained("https://s3.amazonaws.com/models.huggingface.co/bert/monologg/biobert_v1.1_pubmed/config.json")
        encoder_config = AutoConfig.from_pretrained("./weights/biobert_weight/dmis_biobert_large_case/config.json")
        self.encoder = AutoModel.from_pretrained('./weights/biobert_weight/dmis_biobert_large_case/', config=encoder_config)
        #self.encoder = AutoModel.from_pretrained('bert-base-uncased', "monologg/biobert_v1.1_pubmed",config=encoder_config)

        # decoder_config = BertConfig(
        #     num_hidden_layers=6,
        #     vocab_size=50000,
        #     hidden_size=512,
        #     num_attention_heads=8
        # )
        #decoder_config = AutoConfig.from_pretrained("https://s3.amazonaws.com/models.huggingface.co/bert/monologg/biobert_v1.1_pubmed/config.json")
        decoder_config = AutoConfig.from_pretrained("./weights/biobert_weight/dmis_biobert_large_case/config.json")
        decoder_config.is_decoder = True
        print(decoder_config)
        encoder_config.add_cross_attention=True
        decoder_config.add_cross_attention=True

        #self.decoder = AutoModel.from_pretrained('bert-base-uncased', "monologg/biobert_v1.1_pubmed",config=decoder_config)
        self.decoder = AutoModel.from_pretrained('./weights/biobert_weight/dmis_biobert_large_case/', config=decoder_config)

        #self.linear = nn.Linear(512, 50000, bias=False)
        #self.linear = nn.Linear(768, 28996, bias=False)
        #---- Linear Layer for BioBert large case ------
        self.linear = nn.Linear(1024, 58996, bias=False)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        # encoder_hidden_states: [batch_size, max_length, hidden_size]
        #encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
        # out: [batch_size, max_length, hidden_size]
        out = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        out = out.last_hidden_state
#        print("out shape: ",out.shape)
        out = self.linear(out)
        return out

