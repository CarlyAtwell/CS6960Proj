import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from utils import mlp, Net
import os
from gui.filters import apply_filter, FILTERS

### Trains an RL policy on learned reward function


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


# function to train a vanilla policy gradient agent. 
# Altered from Cartpole domain to image filtering domain
def train(hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, reward=None, checkpoint = False, checkpoint_dir = "\."):

    # input dim for policy neural net: size of images
    img_dim = 1024 * 1024
    # num of actions: # of filter choices + stop action
    num_acts = 9

    # make core of policy network
    logits_net = mlp(sizes=[img_dim]+hidden_sizes+[num_acts])

    # make function to compute action distribution
    def get_policy(img):
        # convert img to tensor
        img_tensor = pil_to_tensor(img)
        logits = logits_net(img_tensor)
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
        # make some empty lists for logging.
        batch_imgs = []         # for images
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        img = None               # TODO: first img comes from ? sample from some set of training images?
        done = False            # TODO: when STOP action is chosen instead of a filter to signal that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by applying filter actions to training images with current policy
        while True:

            # save img
            batch_imgs.append(img.copy())

            # act in the environment
            act = get_action(torch.as_tensor(img, dtype=torch.float32))
            
            # put image and selected action in tuple as single step trajectory for predict_reward func
            img_act_pair = [(img, act)]
            
            # predict reward from neural net with reward params trained from pref data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ##### low-key not exactly sure what this is doing/ if we actually need it? ###
            torchified_state = torch.from_numpy(img_act_pair).float().to(device)
            r = reward.predict_reward(torchified_state.unsqueeze(0)).item()
            rew = r

            # now apply filter/action to image
            # saying action index 8 is STOP action, but maybe 0 makes more sense, idk?
            if(act != 8):
                # apply_filter function will return the filtered image
                filter = list(FILTERS)[act] # get filter name using action index
                img = apply_filter(img, filter)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if act == 'STOP':
                # if filtering this image is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables ################################## TODO: set image to next img from training set/folder, not None
                img, done, ep_rews = None, False, []

                # end this batch loop if we have enough of it
                if len(batch_imgs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_imgs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t predicted return: %.3f \t ep_len (gt reward): %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        
        if checkpoint:
            #checkpoint after each epoch
            print("!!!!!! checkpointing policy !!!!!!")
            torch.save(logits_net.state_dict(), checkpoint_dir + '/policy_checkpoint'+str(i)+'.params')
    
    #always at least checkpoint at end of training
    if not checkpoint:
        torch.save(logits_net.state_dict(), checkpoint_dir + '/final_policy.params')
    

if __name__ == '__main__':
    print("testing")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
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
    reward_net = Net()
    reward_net.load_state_dict(torch.load(args.reward_params))
    reward_net.to(device)
    train(lr=args.lr, epochs=args.epochs, reward=reward_net, 
          checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir)