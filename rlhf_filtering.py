import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random
from utils import mlp, Net
import json
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as pyplot
import os

from config import PREF_FILE, OUT_DIR, FULLSET, SAMPLE_SIZE
from gui.filters import apply_filter, FILTER_REVERSE_MAPPING
from tuned_alexnet import TunedAlexNet

# import psutil

def retrieve_explicit_preferences(pref_file):
    '''
    Retrieves explicit preferences from a user using the preference_gui.py to generate a file
    '''

    pil_transform = transforms.Compose([transforms.ToTensor()]) # Use this instead of PILToTensor b/c PILToTensor doesn't normalize to [0,1] float

    base_dir = None
    primary_dir = None
    secondary_dir = None
    extract = None
    with open(pref_file, "r") as f:
        base_dir = f.readline().rstrip()
        primary_dir = f.readline().rstrip()
        secondary_dir = f.readline().rstrip()
        extract = json.load(f)

    print('Num Prefs:', len(extract))
    
    # Iterate extracted to generate trajectories
    # Example format of pref item: ["demo.jpg", 1, ["contrast+10", "contrast+10"], ["color_bal+", "color_bal+"]]
    traj_pairs = []
    traj_labels = []
    for pref in extract:
        # print(pref)
        img_name, label, prim_filts, sec_filts = pref
        
        img_dir = base_dir + '/' + img_name
        
        # base_img = Image.open(img_dir)

        # prim_traj = []
        # prim_state = base_img
        # for action_str in prim_filts:
        #     action = FILTER_REVERSE_MAPPING[action_str]
        #     # Append transition from current state with given action (filter)
        #     prim_traj.append((pil_transform(prim_state).unsqueeze(0), action))

        #     # Do transition
        #     prim_state = apply_filter(prim_state, action_str)
        # # Apply 'none' (stop) action to end of trajectories
        # prim_traj.append((pil_transform(prim_state).unsqueeze(0), FILTER_REVERSE_MAPPING['none']))
        
        # sec_traj = []
        # sec_state = base_img
        # for action_str in sec_filts:
        #     action = FILTER_REVERSE_MAPPING[action_str]
        #     # Append transition from current state with given action (filter)
        #     sec_traj.append((pil_transform(sec_state).unsqueeze(0), action))

        #     # Do transition
        #     sec_state = apply_filter(sec_state, action_str)
        # # Apply 'none' (stop) action to end of trajectories
        # sec_traj.append((pil_transform(sec_state).unsqueeze(0), FILTER_REVERSE_MAPPING['none']))

        # # Append the pair of (s,a) trajectories to the pairs with the corresponding preference label between primary/secondary
        # # TODO: deal with equal preference (2)

        # traj_pairs.append((prim_traj, sec_traj))

        traj_pairs.append((img_dir, prim_filts, sec_filts))

        # traj_labels.append(label)
        #TODO thing i got these backwards b/c of loss function https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
        # If label is 2 we can set it as 0.5 instead
        if label == 2:
            traj_labels.append(0.5)
        else:
            traj_labels.append(1-label)

    # Mem usage
    # process = psutil.Process()
    # print(f'Mem Usage:  {process.memory_info().vms / (1024 ** 3)} GB    {process.memory_info().vms / (1024 ** 2)} MB    {process.memory_info().vms} B')
    print("Extracted explicit prefs from file:", pref_file)
    print(traj_labels)

    return traj_pairs, traj_labels


def gen_pref_traj(pref):
    pil_transform = transforms.Compose([transforms.ToTensor()]) # Use this instead of PILToTensor b/c PILToTensor doesn't normalize to [0,1] float

    img_dir, prim_filts, sec_filts = pref
    
    base_img = Image.open(img_dir)

    prim_traj = []
    prim_state = base_img
    for action_str in prim_filts:
        action = FILTER_REVERSE_MAPPING[action_str]
        # Append transition from current state with given action (filter)
        prim_traj.append((pil_transform(prim_state).unsqueeze(0), action))

        # Do transition
        prim_state = apply_filter(prim_state, action_str)
    # Apply 'none' (stop) action to end of trajectories
    prim_traj.append((pil_transform(prim_state).unsqueeze(0), FILTER_REVERSE_MAPPING['none']))
    
    sec_traj = []
    sec_state = base_img
    for action_str in sec_filts:
        action = FILTER_REVERSE_MAPPING[action_str]
        # Append transition from current state with given action (filter)
        sec_traj.append((pil_transform(sec_state).unsqueeze(0), action))

        # Do transition
        sec_state = apply_filter(sec_state, action_str)
    # Apply 'none' (stop) action to end of trajectories
    sec_traj.append((pil_transform(sec_state).unsqueeze(0), FILTER_REVERSE_MAPPING['none']))

    return (prim_traj, sec_traj)
    

def predict_traj_return(net, traj):
    states = [e[0].to(device) for e in traj]
    actions = [e[1] for e in traj]
    
    actions = torch.tensor(actions).to(device)
    return net.predict_reward(states, actions).item()

# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #We will use a cross entropy loss for pairwise preference learning
    # loss_criterion = nn.CrossEntropyLoss()
    loss_criterion = nn.BCELoss(reduction='sum')
    
    # train reward function using the training data
    # training_inputs gives you a list of pairs of trajectories
    # training_outputs gives you a list of labels (0 if first trajectory better, 1 if second is better)

    loss_history = []

    for iter in range(num_iter):
        
        # Do 5 random so don't run out of mem
        #NOTE: with the fine tuned alexnet I can fit the whole thing now on GPU, but if problems can uncomment the three lines to do sampling

        input_sample = None
        output_sample = None
        if FULLSET:
            input_sample = training_inputs
            output_sample = training_outputs
        else:
            sample_inds = random.sample(range(len(training_inputs)), SAMPLE_SIZE)
            input_sample = [training_inputs[si] for si in sample_inds]
            output_sample = [training_outputs[si] for si in sample_inds]
        

        # input_sample = training_inputs
        # output_sample = training_outputs

        #zero out automatic differentiation from last time
        optimizer.zero_grad()

        # for param in reward_net.parameters():
        #     print(param.grad)

        # predict preferences
        predicted_rewards = []
        i = 0

        # (traj_prim, traj_sec)
        # each traj: [(img_state, action_int),..]
        #for (x,y) in training_inputs:
        #for pref in training_inputs:
        for pref in input_sample:
            x, y = gen_pref_traj(pref)

            #TODO: send to device earlier? is this slow?
            states_x = [e[0].to(device) for e in x]
            actions_x = [e[1] for e in x]
            states_y = [e[0].to(device) for e in y]
            actions_y = [e[1] for e in y]

            # print(states_x)
            # print(actions_x)

            # x = np.array(x)
            # states_x = torch.tensor(states_x).to(device)
            actions_x = torch.tensor(actions_x).to(device)

            # y = np.array(y)
            # states_y = torch.tensor(states_y).float().to(device)
            actions_y = torch.tensor(actions_y).to(device)
            
            # print(actions_x.size(dim=0))

            # print(states_x[0].size())
            # raise NotImplementedError

            x_rew = reward_network.predict_reward(states_x, actions_x)
            y_rew = reward_network.predict_reward(states_y, actions_y)
            # predicted_rewards.append([x_rew, y_rew])

            predicted_rewards.append(torch.exp(x_rew) / (torch.exp(x_rew) + torch.exp(y_rew)))

            # Try to clear memory idk
            # del x
            # del y
            # torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated())

        # compute loss
        # print(predicted_rewards)
        # print(predicted_rewards.size(), training_outputs.size())
        # a = torch.stack(predicted_rewards)
        # b = torch.tensor(training_outputs).float()

        # print(predicted_rewards, training_outputs)

        #loss = loss_criterion(torch.stack(predicted_rewards), torch.tensor(training_outputs).float().to(device))
        loss = loss_criterion(torch.stack(predicted_rewards), torch.tensor(output_sample).float().to(device))

        print("iteration", iter, "bc loss", loss)
        print(torch.cuda.memory_allocated())

        # raise NotImplementedError
        
        #back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        # for param in reward_net.parameters():
        #     print(param.requires_grad, param.grad)

        # raise NotImplementedError

        #perform update on policy parameters
        optimizer.step()

        loss_history.append(float(loss))

        if iter % 5 == 0:
            print("check pointing")
            torch.save(reward_network.state_dict(), f'{OUT_DIR}/rewardnet{iter}.params')

    # After training we save the reward function weights    
    print("check pointing")
    torch.save(reward_network.state_dict(), checkpoint_dir)
    print("finished training")

    return loss_history


def plot_loss_history(loss_hist):
    '''
    Plots the loss history
    '''
    ep = range(len(loss_hist))

    fig, ax = pyplot.subplots(figsize=(14, 10))
    fig.set_tight_layout(True) # Sets nice padding between subplots

    ax.plot(ep, loss_hist, '-b', label = 'training')
    #ax.plot(ep, self.validation_loss_history, '-r', label = 'validation')
    ax.set_title("Loss history")
    #ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")

    fig.savefig(f'{OUT_DIR}/loss_history.png')
    


if __name__=="__main__":

    if not os.path.exists(OUT_DIR):
        print(f"Creating dir '{OUT_DIR}'")
        os.makedirs(OUT_DIR)

    #checkpoint = "./reward.params" #where to save your reward function weights
    checkpoint = f'{OUT_DIR}/rewardnet.params'
    eval_file = f'{OUT_DIR}/eval.txt'



    ###### TODO: create preference data from GUI #####
    traj_pairs, traj_labels = retrieve_explicit_preferences(PREF_FILE) #TODO: alternatively generate synthetic preferences from a user's filtering for experiment 2

    print(traj_labels)
    
    #TODO: hyper parameters that you may want to tweak or change
    num_iter = 30 #150 #300 0.000001; same for 150 I think; 50 was lower 0.0000001
    lr = 0.0001 #5e-4#0.0001
    wd = 3e-3

    # create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #reward_net = Net()
    reward_net = TunedAlexNet()
    print(reward_net)
    reward_net.to(device)

    print("Made net")

    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr)#, weight_decay=wd)

    start = time.time()
    loss_hist = learn_reward(reward_net, optimizer, traj_pairs, traj_labels, num_iter, checkpoint)
    end = time.time()
    print("Elapsed: ", end-start, "s")

    print("Plotting...")
    plot_loss_history(loss_hist)
    print("Plotted")

    #debugging printout
    #we should see higher predicted rewards for more preferred trajectories
    print("performance on training data")
    num_correct = 0
    with open(eval_file, "w") as f:
        for i,pref in enumerate(traj_pairs):
            trajA, trajB = gen_pref_traj(pref)

            arew = predict_traj_return(reward_net, trajA)
            brew = predict_traj_return(reward_net, trajB)
            f.write(f"predicted return trajA {arew}\n")
            f.write(f"predicted return trajB {brew}\n")
            f.write("A " if arew > brew else "B ")
            f.write(f"{traj_labels[i]}\n")
            if traj_labels[i] == 1: #TODO: swapped here b/c 1 means distrib pushed on primary
                f.write("A should be better\n")
                if arew > brew:
                    num_correct += 1
            else:
                f.write("B should be better\n")
                if brew > arew:
                    num_correct += 1

            del trajA
            del trajB

        f.write(f'Correct: {num_correct}/{len(traj_pairs)}\n')
        print(f'Correct: {num_correct}/{len(traj_pairs)}')
        print(traj_labels)