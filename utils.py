import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Conv neural network for the policy

    layers = [nn.Conv2d(3, 6, 5), activation, nn.MaxPool2d(2, 2), 
              nn.Conv2d(6, 18, 5), activation, nn.MaxPool2d(2, 2), 
              nn.Conv2d(18, 32, 5), activation, nn.MaxPool2d(2, 2), nn.Flatten()]
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    # Conv Neural Net for reward function trained on Human Pref data

    def __init__(self):
        super().__init__()
        # defines CNN network architecture. 
        
        # States are of size IMAGE_DIM
        # This network should take in  inputs corresponding to the image pixels and 
        # output logits corresponding to possible filter actions
        
        # flattened image dim
        IMAGE_DIM = 1024*1024
        # max number of filters, if < max # filters, just empty or 0 or filter that doesn't do anything or something
        FILTER_DIM = 9

        self.conv1 = nn.Conv2d(3, 6, 5) # 3 img input channels (RGB), 6 5x5 Conv filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 18, 5)
        self.conv3 = nn.Conv2d(18, 32, 5)
        self.fc1 = nn.Linear(32 * IMAGE_DIM, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, FILTER_DIM)

   
    def predict_reward(self, traj):
        '''calculate cumulative return of trajectory, could be a trajectory with a single element'''
        # should take in a trajectory of image states and filter actions and output a scalar cumulative reward estimate
        # trajectory should be list of tuples of (img, filter_action), then (filtered_img, next_filter_action), etc. max 10

        cumulative_reward = 0
        for img_state, action in traj:
            #this method performs a forward pass through the network
            x = self.pool(F.relu(self.conv1(img_state)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            cumulative_reward += x[action]

        return cumulative_reward



   