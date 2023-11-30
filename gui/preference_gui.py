'''
GUI for selecting pairwise preferences

Load in two folders with same filtered images in same order

Will go through and show each pair (randomizing left/right display) and let user pick preference (or neither)

TODO: consider also having the original image as an option to prefer
TODO: consider more than 2 prefernce options at a time
'''
import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import io
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageEnhance
import time
import os

# Modules
from filters import FILTERS
import gui_utils

### CONFIG #################################################################
default_path = './none.png'

# GUI CONSTANTS ##########
WIN_PAD = 75
IMAGE_SIZE = (800,800)
##########################

############################################################################

left_col = sg.Column(
    [
        [sg.Column(
            [
                # [sg.Image(default_path, key='-IMAGE_LEFT-', size=IMAGE_SIZE)]
                [sg.Button(image_source=default_path, key=f"-IMAGE_LEFT-", image_size=IMAGE_SIZE)]
            ]
        )]
    ],
    element_justification='c',
    key='-COL_LEFT-'
)

right_col = sg.Column(
    [
        [sg.Column(
            [
                [sg.Button(image_source=default_path, key=f"-IMAGE_RIGHT-", image_size=IMAGE_SIZE)]
            ]
        )]
    ],
    element_justification='c',
    key='-COL_RIGHT-'
)

main_layout = [[left_col, right_col]]

window = sg.Window('Preference Generator', main_layout, resizable=True)

class GUI:
    dir_primary = None
    dir_secondary = None

    # List of tuples (orig_filename, out_filename, [list_of_filter_applications])
    filt_hist_primary = []
    filt_hist_secondary = []
    cur_img_ind = -1

    #TODO: figure out what this looks like; prolly do int [filt_chain_0] [filt_chain_1], and int is either 0 or 1 for which was preferred (or 2 for neither)
    preferences = []

    def next_image(self):
        self.cur_img_ind += 1

        primary = self.filt_hist_primary[self.cur_img_ind]
        secondary = self.filt_hist_secondary[self.cur_img_ind]

        # Load in both images
        p_img = Image.open(primary[1])
        s_img = Image.open(secondary[1])

        # nvm filter chain already exists b/c we saved the correct images already
        # but still make that helper will be necessary for the RL portion to reconstruct
        # # Apply filer chain
        # #TODO: add a util in filters.py that takes in a PIL image and list of filters and applies them all sequentially
        # p_img_f = None
        # s_img_f = None

        # Randomize which one gets displayed on L/R
        # Store this as inst variable

        # Display images with gui_utils.rescale() to fit it

        # When user clicks, store the preference

    def main(self):
        while True:
            event, values = window.read(timeout = 50)
            if event == sg.WIN_CLOSED:
                break

        if event == '':
            #TODO:
            # when button clicked, do self.record_preference(), then self.next_image()
            # when last preference recorded, do export of all
            # also have button for early export if anything

            #TODO: also allow rescale of images, same as with thumbnails in other GUI, so can fit better to screen sizes
            pass

        window.close()

############################################################################

if __name__ == "__main__":
    GUI().main()
