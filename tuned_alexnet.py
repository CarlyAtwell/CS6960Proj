import torch
import torch.nn as nn

from torchvision.models import alexnet


class TunedAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super().__init__()

    self.net = alexnet(pretrained=True)
    FILTER_DIM = 9

    # Replace the entire fully connected classifier
    self.net.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, FILTER_DIM),
    )

    # Disable gradient on everything but last layer to 'freeze' the pretrained portion of the model
    for param in self.net.parameters():
      param.requires_grad = False

    # Reenable on last layer
    for param in self.net.classifier.parameters():
      param.requires_grad = True

    # Now shove into cnn and fc layers
    #self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''

    x = self.net.features(x)
    x = self.net.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.net.classifier(x)

    return x
  
  def predict_reward(self, states, actions):
      '''calculate cumulative return of trajectory, could be a trajectory with a single element'''
      # should take in a trajectory of image states and filter actions and output a scalar cumulative reward estimate
      # trajectory should be list of tuples of (img, filter_action), then (filtered_img, next_filter_action), etc. max 10
  
      traj_size = actions.size(dim=0)
      cumulative_reward = 0
      # for img_state, action in traj:
      for i in range(traj_size):
          img_state = states[i]
          action = actions[i]
          x = self.forward(img_state)
          cumulative_reward += x[0][action]
  
      return cumulative_reward