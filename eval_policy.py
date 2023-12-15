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
import json

from tuned_alexnet import TunedAlexNet, TunedPolicyAlexNet

# function to train a vanilla policy gradient agent. 
# Altered from Cartpole domain to image filtering domain
def filter_images(hidden_sizes=[32]):

    #POLICY_PARAMS = './rlhf/policy_checkpoint9.params'
    #POLICY_PARAMS = './reward_checkpoints/tunedalex4full_30iter_0001lr/rewardnet.params'
    #POLICY_PARAMS = './alex_checkpoints/policy_checkpoint29.params'
    #POLICY_PARAMS = './policy_checkpoints_implicit/policy_checkpoint29.params'\
    #POLICY_PARAMS = './reward_checkpoints/tunedalex_victorimplicit/rewardnet.params'
    
    # POLICY_PARAMS = './policy_checkpoints_implicit_limited/policy_checkpoint29.params'
    POLICY_PARAMS = './reward_checkpoints/tunedalex_victorimplicit_limited/rewardnet.params'
    OUTPUT_TXT = 'out.txt'
    #OUTPUT_DIR = './eval/test50'
    #OUTPUT_DIR = './eval/test_alexpolicy'
    #OUTPUT_DIR = './eval/test_policy_implicit'
    #OUTPUT_DIR = './eval/test_rewarddirect_implicit'

    # OUTPUT_DIR = './eval/test_policy_implicit_limited'
    OUTPUT_DIR = './eval/test_rewarddirect_implicit_limited'

    NUM_FILTER = 25#50 # how many you want to filter if loading more than that many
    IMG_DIR = './datasets/test'
    EVAL_IMGS =  [img_file for img_file in listdir(IMG_DIR) if isfile(join(IMG_DIR, img_file)) and img_file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    if len(EVAL_IMGS) > NUM_FILTER:
        EVAL_IMGS = EVAL_IMGS[:NUM_FILTER]

    # input dim for policy neural net: size of images
    img_dim = 1024 * 1024
    # num of actions: # of filter choices + stop action
    num_acts = 9

    # Max actions to take before just injecting a stop command
    MAX_ACTS = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make core of policy network
    #TODO temp try just reward net directly
    #logits_net = mlp(sizes=[img_dim//2]+hidden_sizes+[num_acts]).to(device)
    #logits_net.load_state_dict(torch.load(POLICY_PARAMS))
    logits_net = TunedAlexNet().to(device)
    #logits_net = TunedPolicyAlexNet().to(device)
    logits_net.load_state_dict(torch.load(POLICY_PARAMS))

    # make function to compute action distribution
    def get_policy(img):
        # convert img to tensor
        #img_tensor = pil_to_tensor(img)
        logits = logits_net(img)
        return Categorical(logits=logits)

    # make action/filter selection function (outputs int actions, sampled from policy)
    def get_action(img):
        return get_policy(img).sample().item()

    # for training policy
    def filter_img(img_file):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # reset episode-specific variables
        pil_transform = transforms.Compose([transforms.ToTensor()]) # Use this instead of PILToTensor b/c PILToTensor doesn't normalize to [0,1] float

        img_pil = Image.open(IMG_DIR + '/' + img_file) #pil_transform(Image.open(TRAIN_DIR + '/' + img_file)).unsqueeze(0).to(device) #TODO: need to do unsqeeze(0)? and to(device) ?
        img_tensor = pil_transform(img_pil).unsqueeze(0).to(device)
        episode_acts = 0
        action_traj = []
        # collect experience by applying filter actions to training images with current policy
        while True:
            # act in the environment
            act = get_action(img_tensor)
            episode_acts += 1

            if((act != 8) and (episode_acts < MAX_ACTS)):
                action_traj.append(FILTER_MAPPING[act])
                # apply_filter function will return the filtered image
                filter = FILTER_MAPPING[act] #list(FILTERS)[act] # get filter name using action index
                img_pil = apply_filter(img_pil, filter)
                img_tensor = pil_transform(img_pil).unsqueeze(0).to(device)
            else:
                break

        return img_pil, action_traj

    with torch.no_grad():
        print(len(EVAL_IMGS), EVAL_IMGS)
        actions = []

        isExist = os.path.exists(f'{OUTPUT_DIR}')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(f'{OUTPUT_DIR}')    
                
        for img_file in EVAL_IMGS:
            # Do the filtering
            print(img_file)

            filtered_img, action_traj = filter_img(img_file)

            filtered_img.save(f'{OUTPUT_DIR}/filt_{img_file}')

            actions.append((img_file, action_traj))

        with open(f'{OUTPUT_DIR}/{OUTPUT_TXT}', "w") as f:
            json.dump(actions, f)
                
    

if __name__ == '__main__':
    filter_images()