'''
Utils for extracting export folders into Python datastructures

Useful for both the GUI and the RL
'''

import os
import json

def extract_folder(path):
    '''
    Extracts items from folder into list of tuple (input_image_name, output_image_name, [list_of_filters])
    '''
    export_name = os.path.basename(path)

    extract = None
    with open(f'{path}/{export_name}.txt', "r") as f:
        extract = json.load(f)

    return extract