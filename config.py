# Current user preference file to use if doing explicit preferences (Experiment 1)
# Generated from ./gui/preferences_gui.py
PREF_FILE = './gui/230Pref_1newdirs.txt'
#CHECKPOINT_FILE = './reward_checkpoints/reward_230Pref_1.params'
CHECKPOINT_FILE = './reward_checkpoints/reward_230Pref_1_150iter.params'
EVAL_FILE = './reward_checkpoints/reward_230Pref_1_150iter.txt'

# We want to not show some portion of the images so we can use them to both evaluate the reward network's fit and to use at the end after the policy generates images
# so we can compare against which image the user liked, and if what the network spits out for that image is somewhat similar to the user preference (?)
# For the case where we have users do the filtering, this is even better b/c we can directly compare it against their filtered output and ask how good a job the policy did
# at getting a similar result (7 point scale)
VALIDATION_SET = 30