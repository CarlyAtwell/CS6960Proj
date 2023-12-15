import random
import time
import os
import json

import gui.filters as filters
import gui.data_utils as data_utils


# If we have 8 filters that can happen up to 10 times, that is 8^10=1,073,741,824 possible filterings, and that's just the 10-length - also have 9-length, 8-length, etc...
# But overall 8^10 dominates so roughly 1 billion possibilities
# Obviously we don't want a single filtering operation to generate 1 billion preferences, we would train forever and the network at that point is probably not going to learn anything
# useful from that data
# Thus instead, we will just generate a sample of X possible other filtering combinations, of each length, for each preference
# Given stochasticity, should probably end up learning something useful..?

VALID_FILTS = filters.FILTER_MAPPING[:-1]
MAX_FILT_LENGTH = 10
FILT_LENGTH_LIMITED = True # if should limit to the lenghth of the trajectory+1; maybe helps STOP action from being so dominant in reward signal

NUM_GENERATIONS = 3 # How many examples to generate of each trajectory length

def get_permutation(length):
    '''
    Random permutation with replacement

    Duplicates pretty much impossible
    '''
    return [random.choice(VALID_FILTS) for _ in range(length)]

def gen_pairwise_prefs(filter_folder_dir):
    # [ ["0000000.jpg", "out_0000000.jpg", ["color_bal+", "contrast+10", "contrast+10", "bright-10"]], ...]
    filt_hist = data_utils.extract_folder(filter_folder_dir)
    print(f"Receive {len(filt_hist)} Filter Histories")
    preferences = []
    dupes_removed = 0

    for img_file, _, traj in filt_hist:
        other_traj = [[]] # for the choise of applying no filters (IE, applying the STOP operation immediately)
        max_traj_length = len(traj) + 2 if FILT_LENGTH_LIMITED else MAX_FILT_LENGTH #+2 is arbitrary; to show it don't do more filtering, but not exceedingly more
        for i in range(max_traj_length):
            length = i + 1
            for n in range(NUM_GENERATIONS):
                other_traj.append(get_permutation(length))

        processed_traj = []
        # Remove duplicates if any
        [processed_traj.append(t) for t in other_traj if t not in processed_traj and t != traj]
        dupes_removed += len(other_traj) - len(processed_traj)
        
        # for t in other_traj:
        #     if t != traj:
        #         processed_traj.append(t)

        # Create preference of traj to all in processed_traj
        for pt in processed_traj:
            # Always do '0' pref bc put traj on left
            preferences.append((img_file, 0, traj, pt))

    print(f"Removed {dupes_removed} duplicates")
    
    return preferences


def gen_traj_file(filter_folder_dir, export_name):
    # Generate preferences
    preferences = gen_pairwise_prefs(filter_folder_dir)
    print(f"Gen {len(preferences)} Implicit Preferences")

    # Export

    # Default name if none
    if not export_name:
        export_name = f'prefs_{time.time()}' 
    # Outputs json
    outfile = f'./{export_name}.txt'
    with open(outfile, "w") as f:
        # Write out which folders got preferences from
        # Write which folder get base images from
        f.write(os.path.abspath(os.path.dirname(filter_folder_dir)) + '\n')
        # First is primary, second is secondary
        f.write('unused\n')
        f.write('unused\n')

        # Write out JSON preferences
        json.dump(preferences, f)

    print("Export complete:", outfile)


############################################################################

if __name__ == "__main__":
    FILTER_FOLDER = './datasets/manual/victor_manual'
    EXPORT_NAME = 'victor_manual_implicit_limited'
    # FILTER_FOLDER = './datasets/manual_mini/victor0'
    # EXPORT_NAME = 'victor_mini_implicit'

    gen_traj_file(FILTER_FOLDER, EXPORT_NAME)

