import numpy as np
import torch
from torch.nn.modules import GRU

def build_net(input_dict, activation):
    gru_hidden_size=256
    out_size=64

    input_dim = np.sum([np.prod(curr_input.shape) for curr_input in input_dict.values()])
    
    layers = []
    layers.append(GRU(input_size=input_dim, hidden_size=gru_hidden_size, num_layers=1)) 
    layers.append(torch.nn.Linear(gru_hidden_size, out_size))
    layers.append(activation())
    net = torch.nn.Sequential(*layers)
    info = dict()

    return net, info

class gru_net(torch.nn.Module):
    def __init__(self, input_dim, gru_hidden_size=256, num_layers=1):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_size
        self.gru = GRU(input_size=input_dim, hidden_size=gru_hidden_size, num_layers=num_layers)

    def forward(self, x, h=None):
        return self.gru(x, h)