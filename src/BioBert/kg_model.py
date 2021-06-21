import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(nn.Module):
    def __init__(self, num_node_features):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 768)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(x)
        # print(edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x

class transformers_model(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        encoder_config = AutoConfig.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed/config.json")
        
        self.encoder = AutoModel.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed",config=encoder_config)

        decoder_config = AutoConfig.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed/config.json")
        
        self.decoder = AutoModel.from_pretrained("weights/biobert_weight/biobert_v1.1_pubmed",config=decoder_config)
        decoder_config.is_decoder = True
        encoder_config.add_cross_attention=True
        decoder_config.add_cross_attention=True
        self.kg_model = Net(num_node_features)

        self.linear = nn.Linear(768, 28996, bias=False)


    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input, kg_input):
        # encoder_hidden_states: [batch_size, max_length, hidden_size]
        #encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
        # out: [batch_size, max_length, hidden_size]
        # kg_hidden []
        # print('encoder', encoder_hidden_states.view(-1,encoder_hidden_states.size(-1)).size())
        kg_hidden_states = self.kg_model(kg_input)
        # print('kg_input', kg_hidden_states.size())


        hidden_states = torch.cat((encoder_hidden_states.squeeze(0), kg_hidden_states),dim=0)
        # print(hidden_states.unsqueeze(0).size())

        out = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=hidden_states.unsqueeze(0))
        out = out.last_hidden_state
        out = self.linear(out)

        return out
