import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from utils import mlp, Net
import os
from os import listdir
from os.path import isfile, join
from gui.filters import apply_filter, FILTERS, FILTER_MAPPING
import torchvision.transforms as transforms 
from PIL import Image
import random
import matplotlib.pyplot as pyplot
import time

from tuned_alexnet import TunedPolicyAlexNet, TunedAlexNet

def plot_loss_history(loss_hist, checkpoint_dir):
    '''
    Plots the loss history
    '''
    ep = range(len(loss_hist))

    fig, ax = pyplot.subplots(figsize=(14, 10))
    fig.set_tight_layout(True) # Sets nice padding between subplots

    ax.plot(ep, loss_hist, '-b', label = 'training')
    ax.set_title("Loss history")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")

    fig.savefig(f'{checkpoint_dir}/loss_history.png')

def plot_return_history(ret_hist, checkpoint_dir):
    '''
    Plots the loss history
    '''
    ep = range(len(ret_hist))

    fig, ax = pyplot.subplots(figsize=(14, 10))
    fig.set_tight_layout(True) # Sets nice padding between subplots

    ax.plot(ep, ret_hist, '-b', label = 'training')
    ax.set_title("Predicted Return History")
    ax.set_ylabel("Predicted Return")
    ax.set_xlabel("Epochs")

    fig.savefig(f'{checkpoint_dir}/return_history.png')

### Trains an RL policy on learned reward function

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


# function to train a vanilla policy gradient agent. 
# Altered from Cartpole domain to image filtering domain
def train(hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=25, reward=None, checkpoint = False, checkpoint_dir = "\."):

    # input dim for policy neural net: size of images
    img_dim = 1024 * 1024
    # num of actions: # of filter choices + stop action
    num_acts = 9

    # Max actions to take before just injecting a stop command
    MAX_ACTS = 10

    # get training images
    TRAIN_DIR = './datasets/train'
    TRAIN_IMGS =  [img_file for img_file in listdir(TRAIN_DIR) if isfile(join(TRAIN_DIR, img_file)) and img_file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    next_img_batch = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make core of policy network
    #logits_net = mlp(sizes=[img_dim//2]+hidden_sizes+[num_acts]).to(device)
    logits_net = TunedPolicyAlexNet().to(device)

    # make function to compute action distribution
    def get_policy(img):
        # convert img to tensor
        #img_tensor = pil_to_tensor(img)
        logits = logits_net(img)
        # print(logits)
        return Categorical(logits=logits)

    # make action/filter selection function (outputs int actions, sampled from policy)
    def get_action(img):
        return get_policy(img).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(img, act, weights):
        logp = get_policy(img).log_prob(act)
        return -(logp * weights).mean()
    

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        
        # Sample batch_size random images from the TRAIN_IMGS
        curr_img = 0
        batch_files = random.sample(TRAIN_IMGS, batch_size)


        # make some empty lists for logging.
        batch_imgs = []         # for images
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # reset episode-specific variables
        pil_transform = transforms.Compose([transforms.ToTensor()]) # Use this instead of PILToTensor b/c PILToTensor doesn't normalize to [0,1] float
        #img_file = TRAIN_IMGS[next_img]       # first img for this epoch comes from set of training images
        img_file = batch_files[curr_img]       # first img for this epoch comes from set of training images
        img_pil = Image.open(TRAIN_DIR + '/' + img_file)
        img_tensor = pil_transform(img_pil).unsqueeze(0).to(device)
        ep_rews = []                    # list for rewards accrued throughout ep

        episode_acts = 0
        imgs_filtered = 0

        # collect experience by applying filter actions to training images with current policy
        while True:

            # save img
            #batch_imgs.append(img_pil.copy())
            batch_imgs.append(img_tensor.clone().squeeze()) # we squeeze here b/c we need to then combine batch_imgs into one batch tensor

            # act in the environment
            act = get_action(img_tensor)

            act_tensor = torch.tensor(act).to(device)

            episode_acts += 1
            
            # predict reward from neural net with reward params trained from pref data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Stack these so that they are a list; can probably do unsqueeze(0) instead too
            r = reward.predict_reward(torch.stack((img_tensor,)), torch.stack((act_tensor,))).item()
            rew = r

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            # now apply filter/action to image
            # saying action index 8 is STOP action, but maybe 0 makes more sense, idk?
            if((act != 8) and (episode_acts < MAX_ACTS)):
                # apply_filter function will return the filtered image
                filter = FILTER_MAPPING[act] #list(FILTERS)[act] # get filter name using action index
                img_pil = apply_filter(img_pil, filter)
                img_tensor = pil_transform(img_pil).unsqueeze(0).to(device)

            # Put a max number of applications before we cut it off so it doesn't apply filters forever in a bad case
            else:
                # if filtering this image is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # Next image
                curr_img += 1
                if curr_img >= batch_size:
                    break

                img_file = batch_files[curr_img]
                img_pil = Image.open(TRAIN_DIR + '/' + img_file)
                img_tensor = pil_transform(img_pil).unsqueeze(0).to(device)
                done, ep_rews = False, []

                episode_acts = 0

                imgs_filtered += 1

        # take a single policy gradient update step
        optimizer.zero_grad()

        obs=torch.stack(batch_imgs)
        acts=torch.as_tensor(batch_acts, dtype=torch.int32).to(device)
        weights=torch.as_tensor(batch_weights, dtype=torch.float32).to(device)

        batch_loss = compute_loss(img=obs,
                                  act=acts,
                                  weights=weights
                                  )
        batch_loss.backward()
        optimizer.step()

        return batch_loss, batch_rets, batch_lens

    # training loop
    loss_hist = []
    pred_ret_hist = []
    for i in range(epochs):
        print(next_img_batch)
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        next_img_batch += batch_size
        print('epoch: %3d \t loss: %.3f \t predicted return: %.3f \t ep_len (gt reward): %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        
        loss_hist.append(float(batch_loss))
        pred_ret_hist.append(float(np.mean(batch_rets)))
        
        if checkpoint:
            if (i == epochs-1) or i % 5 == 0:
                #checkpoint after each epoch
                print("!!!!!! checkpointing policy !!!!!!")
                torch.save(logits_net.state_dict(), checkpoint_dir + '/policy_checkpoint'+str(i)+'.params')
    
    #always at least checkpoint at end of training
    if not checkpoint:
        torch.save(logits_net.state_dict(), checkpoint_dir + '/final_policy.params')

    print("Plotting...")
    plot_loss_history(loss_hist, checkpoint_dir)
    plot_return_history(pred_ret_hist, checkpoint_dir)
    print("Plotted")
    

if __name__ == '__main__':
    print("testing")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    #parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=15) #NOTE: if this is too high then we run out of GPU memory so keep it small
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='\.')
    parser.add_argument('--reward_params', type=str, default='', help="parameters of learned reward function")
    args = parser.parse_args()
    
    #create checkpoint directory if it doesn't already exist
    isExist = os.path.exists(args.checkpoint_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(args.checkpoint_dir)

    #pass in parameters for trained reward network and train using that
    print("training on learned reward function")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #reward_net = Net()
    reward_net = TunedAlexNet()
    reward_net.load_state_dict(torch.load(args.reward_params))
    reward_net.to(device)

    start = time.time()
    train(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, reward=reward_net, 
          checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir)
    end = time.time()
    print("Elapsed: ", end-start, "s")