# PIL to tensor
https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/

# Stuff to do?
For our reward func we would need to input a state and action, and get the reward for that transitions

I looked at some stuff online and it looks like one approach to make things simpler is instead of feeding in a state and an action, and getting a single output of reward, you just feed a state, and you have N outputs one for each of your potential actions. Therefore to get the reward for (s,a) you just feed in s and query the N outputs for the specific neuron corresponding to taking action a. This will make the CNN thing a lot easier I feel to implement so prolly a good option for our reward func
- https://user.ceng.metu.edu.tr/~emre/resources/courses/AdvancedDL_Spring2017/DQN_Muhammed.pdf slide 18 shows these and says this is the 'less naive' approach

For policy we take in a state and get an action, so it should pretty much be the same architecture

For the CNN, we have 3 input channels, and I'm thinking maybe each channel we split into 1-2 convolution channels for each of our N actions? So like for 5 possible filters, we would do 5 * 2 * 3 = 30 channels in the next layer or something; I think that way we could be able to get each filter to kind of be able to have its own dedicated convolution kernels so maybe it improves its ability to learn something specific to identify how good each one would be. But maybe that's too many weights so we should do less? Idk what rules of thumb there are for choosing how many channels to break up into for CNN

# Dataset
https://www.kaggle.com/datasets/hshukla/landscapes