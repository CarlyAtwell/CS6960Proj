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
import json

# Modules
from filters import FILTERS
import gui_utils
import data_utils

### CONFIG #################################################################
default_path = './none.png'

# GUI CONSTANTS ##########
WIN_PAD = 75
IMAGE_SIZE = (800,800)
##########################

############################################################################

header_bar = sg.Column(
    [
        [
            sg.Button("Load Primary", key='-LOAD_PRIMARY-'), sg.Button("Load Secondary", key='-LOAD_SECONDARY-'), sg.Button("Start", key='-START-') #, sg.Push(), sg.Button("Reset", key='-RESET-')
        ],
        [
            sg.Text("Primary: <none>", key='-LABEL_PRIMARY-')
        ],
        [
            sg.Text("Secondary: <none>", key='-LABEL_SECONDARY-')
        ],
        [
            sg.Text("Export: <none>", key='-LABEL_EXPORT-', visible=False)
        ],
        [
            sg.Text(f"Image Size: {IMAGE_SIZE}", key='-LABEL_IMAGESIZE-'), sg.Button("-", key='-SIZE_M-'), sg.Button("+", key='-SIZE_P-')
        ]
    ]
)

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

mid_col = sg.Column(
    [
        [sg.Column(
            [
                [sg.Button("Equal", key=f"-IMAGE_NONE-", size=(10,10))]
            ]
        )]
    ],
    element_justification='c',
    key='-COL_MID-'
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

main_layout = [[header_bar],[left_col, mid_col, right_col]]

window = sg.Window('Preference Generator', main_layout, resizable=True)

class GUI:
    image_size = IMAGE_SIZE

    dir_primary = None
    dir_secondary = None
    active = False

    # List of tuples (orig_filename, out_filename, [list_of_filter_applications])
    filt_hist_primary = []
    filt_hist_secondary = []
    cur_img_ind = -1
    num_total_imgs = 0
    image_shuffle = (0, 1) # tells which image is on which side (L/R); 0 for primary, 1 for secondary

    # (orig_image_file int [filt_chain_0] [filt_chain_1]), and int is either 0 or 1 for which was preferred (or 2 for neither)
    preferences = []

    def reset(self):
        self.cur_img_ind = -1
        self.preferences = []

    def set_active(self):
        self.active = self.dir_primary and self.dir_secondary

    def record_preference(self, pref):
        self.preferences.append((self.filt_hist_primary[self.cur_img_ind][0], pref, self.filt_hist_primary[self.cur_img_ind][2], self.filt_hist_secondary[self.cur_img_ind][2]))

    def next_image(self):
        '''
        Goes to next image

        If all images assigned, returns True
        '''
        self.cur_img_ind += 1

        if self.cur_img_ind >= self.num_total_imgs:
            return True
        
        # Randomize which one gets displayed on L/R
        self.image_shuffle = (0,1) if np.random.randint(0, 2) == 0 else (1, 0)

        self.set_image()
        # window['-LABEL_PRIMARY-'].update(f"DEBUG {self.image_shuffle}")
        return False
    
    def set_image(self):
        primary = self.filt_hist_primary[self.cur_img_ind]
        secondary = self.filt_hist_secondary[self.cur_img_ind]

        # Load in both images, scaled to window
        p_img = gui_utils.rescale(Image.open(self.dir_primary + "/" + primary[1]), self.image_size)
        s_img = gui_utils.rescale(Image.open(self.dir_secondary + "/" + secondary[1]), self.image_size)

        imgs = [p_img, s_img]
        
        window['-IMAGE_LEFT-'].update(image_data=gui_utils.get_pil_data(imgs[self.image_shuffle[0]]), visible=True)
        window['-IMAGE_RIGHT-'].update(image_data=gui_utils.get_pil_data(imgs[self.image_shuffle[1]]), visible=True)
    
    def export(self):
        export_name = f'prefs_{time.time()}'
        # Outputs json
        outfile = f'./{export_name}.txt'
        with open(outfile, "w") as f:
            # Write out which folders got preferences from
            # First is primary, second is secondary
            f.write(self.dir_primary + '\n')
            f.write(self.dir_secondary + '\n')

            # Write out JSON preferences
            json.dump(self.preferences, f)

        print("Export complete:", outfile)

        return outfile


    # When user clicks, store the preference

    def main(self):
        while True:
            event, values = window.read(timeout = 50)
            if event == sg.WIN_CLOSED:
                break

            if event == '-LOAD_PRIMARY-':
                dir = sg.popup_get_folder('Open', no_window = True)
                print('OPENED:', dir)

                if dir:
                    self.dir_primary = dir

                    self.filt_hist_primary = data_utils.extract_folder(dir)
                    self.preferences = [] # reset preferences when load new
                    self.set_active()

                    self.num_total_imgs = len(self.filt_hist_primary)

                    window['-LABEL_PRIMARY-'].update(f"Primary: {self.dir_primary}")
            
            if event == '-LOAD_SECONDARY-':
                dir = sg.popup_get_folder('Open', no_window = True)
                print('OPENED:', dir)

                if dir:
                    self.dir_secondary = dir

                    self.filt_hist_secondary = data_utils.extract_folder(dir)
                    self.preferences = [] # reset preferences when load new
                    self.set_active()

                    window['-LABEL_SECONDARY-'].update(f"Secondary: {self.dir_secondary}")

            if self.active and event == '-START-':
                self.next_image()

            if self.active and event == '-RESET-':
                self.reset()

            if self.active and event == '-SIZE_M-':
                # Don't resize if too small
                if self.image_size[0] > 100:
                    self.image_size = (self.image_size[0] - 100, self.image_size[1] - 100)
                    self.set_image()

                    # Set label
                    window['-LABEL_IMAGESIZE-'].update(f'Image Size: {self.image_size}')

            if self.active and event == '-SIZE_P-':
                # Don't resize if too large
                if self.image_size[0] < 1500:
                    self.image_size = (self.image_size[0] + 100, self.image_size[1] + 100)
                    self.set_image()

                    # Set label
                    window['-LABEL_IMAGESIZE-'].update(f'Image Size: {self.image_size}')

            if self.active and '-IMAGE' in event:
                pref = None
                if event == '-IMAGE_LEFT-':
                    # Prefer left
                    pref = self.image_shuffle[0]
                elif event == '-IMAGE_RIGHT-':
                    # Prefer right
                    pref = self.image_shuffle[1]
                elif event == '-IMAGE_NONE-':
                    # No preference
                    pref = 2

                self.record_preference(pref)
                
                if self.next_image():
                    # Finished everything, do export
                    print("EXPORTING")
                    outfile = self.export()

                    self.active = False

                    window['-LABEL_EXPORT-'].update(f"Export: {outfile}", visible=True)

            #TODO: also have button for early export if anything

        window.close()

############################################################################

if __name__ == "__main__":
    GUI().main()
