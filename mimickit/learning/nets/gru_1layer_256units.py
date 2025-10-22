import numpy as np
import torch
from torch.nn.modules import GRU

def build_net(input_dict, activation):
    
    input_dim = np.sum([np.prod(curr_input.shape) for curr_input in input_dict.values()])
    
    net = GRU(input_size=input_dim, hidden_size=256, num_layers=1)
    info = dict()

    return net, info
