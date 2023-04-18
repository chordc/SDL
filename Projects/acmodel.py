import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

class ACModel(nn.Module):
    #add actor_critic model to rnn.py
    #rnn.py refer to github link: https://github.com/WassimTenachi/PhySO/tree/main/physo/learn
    def __init__(self, 
                 input_size, 
                 output_size,
                 hidden_size,
                 n_layers      = 1,
                 input_dense   = None,
                 stacked_cells = None,
                 is_lobotomized = False,):
        super().__init__()

       # Input dense layer
        self.input_size  = input_size
        self.hidden_size = hidden_size
        if input_dense is None:
            input_dense = torch.nn.Linear(self.input_size, self.hidden_size)
        self.input_dense = input_dense

        # Define lstm cells
        self.n_layers= n_layers
        if stacked_cells is None:
            stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size  = self.hidden_size,
                                                                    hidden_size = self.hidden_size)
                                for _ in range(self.n_layers) ])

        self.stacked_cells = stacked_cells
        self.output_size= output_size
        

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            F.relu(),
            nn.Linear(64, self.output_size)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            F.relu(),
            nn.Linear(64, 1)
        )

        if output_dense is None:
            output_dense=torch.nn.Linear(self.hidden_size,self.output_size)
        self.output_dense=output_dense
        self.output_activation = lambda x: -torch.nn.functional.relu(x) # Mapping output to log(p)
                                 #lambda x: torch.nn.functional.softmax(x, dim=1)
                                 #torch.sigmoid
        # --------- Annealing param ---------
        self.logTemperature = torch.nn.Parameter(1.54*torch.ones(1), requires_grad=True)
        # lobotomization
        self.is_lobotomized = is_lobotomized
    
    
    def get_zeros_initial_state(self, batch_size):
        zeros_initial_state = torch.zeros(self.n_layers, 2, batch_size, self.hidden_size, requires_grad=False,)
        return zeros_initial_state


    def forward(self, input_tensor, states):
        # Input dense layer
        hx = self.input_dense(input_tensor)
        # layer norm + activation
        new_states = []
        for i in range(self.n_layers):
            hx, cx = self.stacked_cells[i](hx, (states[i,0,:,:], states[i,1,:,:]))
            new_states.append(torch.stack([hx,cx]))
        

        x = self.actor(hx)
        #dist = Categorical(logits=F.log_softmax(x, dim=1))
        #res=x+self.logTemperature
        #res=self.output_dense(hx)+self.logTemperature
        #res = self.output_activation(res)
        #res=F.log_softmax(x, dim=1)
        res=x

        # Probabilities from random number generator
        if self.is_lobotomized:
            res= torch.log(torch.rand(res.shape))

        x = self.critic(hx)
        value = x.squeeze(1)
        
        out_states = torch.stack(new_states)
        return res,value, out_states
    
    def count_parameters (self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params
