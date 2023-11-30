import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network for the policy
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    ##TODO you can code this up using whatever methods you want, 
    # but to work with the rest of the code make sure you
    # at least implement the predict_reward function below

    def __init__(self):
        super().__init__()
        #TODO define network architecture. 
        
        # Hint, states  are X-dimensional -- DIM OF IMAGES + FILTER DIM
        # This network should take in  inputs corresponding to the image pixels and applied filters
        
        # flattened image dim
        IMAGE_DIM = None
        # max number of filters, if < max # filters, just empty or 0 or filter that doesn't do anything or something
        FILTER_DIM = 6

        self.fc1 = nn.Linear(4,8)
        self.fc2 = nn.Linear(8,8)
        self.fc3 = nn.Linear(8,1)

   
    def predict_reward(self, traj):
        '''calculate cumulative return of trajectory, could be a trajectory with a single element'''
        #TODO should take in a image + filters and output a scalar cumulative reward estimate

        cumulative_reward = 0
        for state in traj:
            #this method performs a forward pass through the network
            x = F.relu(self.fc1(state))
            x = self.fc2(x)
            x = self.fc3(x)
            cumulative_reward += x

        return cumulative_reward



   