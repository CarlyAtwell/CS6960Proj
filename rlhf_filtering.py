import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random
from utils import mlp, Net

## TODO: Import stuff for gui


def predict_traj_return(net, traj):
    traj = np.array(traj)
    traj = torch.from_numpy(traj).float().to(device)
    return net.predict_reward(traj).item()

# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #We will use a cross entropy loss for pairwise preference learning
    loss_criterion = nn.CrossEntropyLoss()
    
    # train reward function using the training data
    # training_inputs gives you a list of pairs of trajectories
    # training_outputs gives you a list of labels (0 if first trajectory better, 1 if second is better)
    

    for iter in range(num_iter):
        #zero out automatic differentiation from last time
        optimizer.zero_grad()

        # predict preferences
        predicted_rewards = []
        i = 0
        for (x,y) in training_inputs:

            x = np.array(x)
            x = torch.from_numpy(x).float().to(device)

            y = np.array(y)
            y = torch.from_numpy(y).float().to(device)

            x_rew = reward_network.predict_reward(x)
            y_rew = reward_network.predict_reward(y)
            predicted_rewards.append([x_rew, y_rew])

        # compute loss
        loss = loss_criterion(torch.tensor(predicted_rewards, requires_grad=True), torch.tensor(training_outputs))
        print("iteration", iter, "bc loss", loss)
        
        #back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        #perform update on policy parameters
        optimizer.step()

    # After training we save the reward function weights    
    print("check pointing")
    torch.save(reward_network.state_dict(), checkpoint_dir)
    print("finished training")





if __name__=="__main__":


    ###### TODO: create preference data from GUI #####
    traj_pairs, traj_labels = None
    
    #TODO: hyper parameters that you may want to tweak or change
    num_iter = 100
    lr = 0.00001
    checkpoint = "./reward.params" #where to save your reward function weights

    # create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)


    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr)

    learn_reward(reward_net, optimizer, traj_pairs, traj_labels, num_iter, checkpoint)


    #debugging printout
    #we should see higher predicted rewards for more preferred trajectories
    print("performance on training data")
    for i,pair in enumerate(traj_pairs):
        trajA, trajB = pair
        print("predicted return trajA", predict_traj_return(reward_net, trajA))
        print("predicted return trajB", predict_traj_return(reward_net, trajB))
        if traj_labels[i] == 0:
            print("A should be better")
        else:
            print("B should be better")

