import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import io
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageEnhance

# Modules
import utils
import filters

#### GUI HELPERS ###########################################################
def set_size(element, size):
    # Only work for sg.Column when `scrollable=True` or `size not (None, None)`
    options = {'width':size[0], 'height':size[1]}
    if element.Scrollable or element.Size!=(None, None):
        element.Widget.canvas.configure(**options)
    else:
        element.Widget.pack_propagate(0)
        element.set_size(size)

def gen_tile(event, i, curr_img):
    img_bytes = get_img_bytes(curr_img)
    return [
        [sg.Frame(event + str(i), layout=[
            [sg.Image(data=img_bytes, size=(100,100))],
            [sg.Button("Remove", key=f'-REMOVE-{i}')]
        ])]
    ]

def gen_filter_group(group):
    name = group[0]
    filts = group[1]

    tiles = []

    for key in filts:
        tiles.append(sg.Column(
            [[sg.Button(image_source=default_path, key=f"-THUMB-{key}", image_size=THUMBNAIL_SIZE)], [sg.Text(key)]],
            element_justification='c'))

    return sg.Frame(title=name, layout=[tiles])
############################################################################

### CONFIG #################################################################
default_path = './none.png'

# GUI CONSTANTS ##########
SIZES = {
    '-MAIN-': (800, 1000),
    '-SIDE-': (200, 600)
}

WIN_PAD = 75

THUMBNAIL_SIZE = (200,200)
MAIN_SIZE = (800,800)
##########################

# (name, func_ptr, args_dict)
FILTERS = {
    'bright+10': (filters.brightness, {'val': 1.1}),
    'bright-10': (filters.brightness, {'val': 0.9}),
    'color_bal+': (filters.color_balance, {'val': 2.0}),
    'color_bal-': (filters.color_balance, {'val': 0.5}),
    'contrast+10': (filters.contrast, {'val': 1.1}),
    'contrast-10': (filters.contrast, {'val': 0.9}),
    'sharpness+': (filters.sharpness, {'val': 2.0}),
    'sharpness-': (filters.sharpness, {'val': 0.0})
    #TODO: channel mixing
}

FILTER_GROUPS = [
    ('Brightness', ['bright-10', 'bright+10']),
    ('Color Balance', ['color_bal-', 'color_bal+']),
    ('Contrast', ['contrast-10', 'contrast+10']),
    ('Sharpness', ['sharpness-', 'sharpness+']),
]

PREVIEWS = [
    './red.png', './green.png', './blue.png'
]
############################################################################

main_col = sg.Column(
    [
        [sg.Column([
            [sg.Button("Load", key='-LOAD-'), sg.Button("Folder", key='-FOLDER-')]
            ], justification='l')],
        
        [sg.Image(default_path, key = '-MAIN_IMAGE-', size=MAIN_SIZE)],
    ], 
    element_justification='c', 
    key="-MAIN-", 
    size=SIZES['-MAIN-'])

filter_col = sg.Column([
        [gen_filter_group(group)] for group in FILTER_GROUPS
    ])

main_layout = [[main_col, filter_col]]

window = sg.Window('Image Editor', main_layout, resizable=True)

# PIL
def get_pil_data(img):
    """Generate image data using PIL
    """
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

class FilterHistory:
    filepath = None
    filename = None
    base_img = None
    images = [] # PIL images
    filters = [] # Filter history
    ind = -1

    def __init__(self, path, filename) -> None:
        self.filename = filename
        self.filepath = path + '/' + filename

        self.load()

    def clean(self):
        '''
        To preserve space

        Purges all image data

        Use load() to reload data
        '''
        self.images = []
        self.base_img = None

    def get_cur(self):
        '''
        Gets the current image based on index position
        '''
        if self.ind == -1:
            return self.base_img
        return self.images[self.ind]

    def load(self):
       '''
       Reloads images from the current filter pool
       '''
       self.base_img = Image.open(self.filepath)

       cur_ind = self.ind

       self.ind = -1 # set back to go through and redo everything
       for filter in self.filters:
           self.apply(filter)
           
       # Reset active
       self.ind = cur_ind
    #    cur_img = self.base_img
    #    for filter in self.filters:
    #        func, args = FILTERS[filter]

    #        cur_img = func(cur_img, **args)
    #        self.images.append(cur_img)
    #        self.ind += 1

    def apply(self, filter):
        '''
        Applies a filter
        
        If index is not at the end, removes everything after it (everything that was undone)
        '''
        # Remove stuff that was undone
        filters = filters[:self.ind+1]
        images = images[:self.ind+1]

        cur_img = images[self.ind] if self.ind != -1 else self.base_img

        func, args = FILTERS[filter]
        cur_img = func(cur_img, **args)
        self.images.append(cur_img)
        self.ind += 1

    def set_active(self, ind):
        '''
        Change which of the history is active
        '''
        self.ind = ind

class GUI:
    cur_dir = None

    cur_img_ind = -1
    pil_cur_img = None
    pil_thumbnail = None
    filt_hist = []

    main_img_size = MAIN_SIZE #TODO: work this out from aspect ratio and main max size (?)
    curr_win_size = (0,0)

    def load_img(self):
        self.pil_cur_img = Image.open(self.cur_dir + '/' + self.filt_hist[self.cur_img_ind][0])
        self.pil_thumbnail = self.pil_cur_img.resize(THUMBNAIL_SIZE)

        # Set preview
        window['-MAIN_IMAGE-'].update(data=get_pil_data(self.pil_cur_img), size=self.main_img_size)

        # Regen thumbs
        self.regen_thumbs()

    def regen_thumbs(self):
        # Apply all filters to thumnail and update previews
        self.pil_thumbnail = self.pil_cur_img.resize(THUMBNAIL_SIZE)
        for key, value in FILTERS.items():
            func, args = value

            thumb_filtered = func(self.pil_thumbnail, **args)

            window[f'-THUMB-{key}'].update(image_data=get_pil_data(thumb_filtered), visible=True)

    def main(self):
        while True:
            event, values = window.read(timeout = 50)
            if event == sg.WIN_CLOSED:
                break

            # TODO: here can handle resize events
            if not utils.vec_equal2d(self.curr_win_size, window.size):
                self.curr_win_size = window.size
                #set_size(window['-SIDE-'], (SIZES['-SIDE-'][0], curr_win_size[1]-WIN_PAD))
            
            if "-THUMB-" in event:
                print("FILTER:", event)
                # Apply filter, update thumbnails
                # Retrieve filter key
                key = event[7:]
                func, args = FILTERS[key]

                # Add to history for this image
                self.filt_hist[self.cur_img_ind][1].append(key)
                # Update current image
                self.pil_cur_img = func(self.pil_cur_img, **args)
                window['-MAIN_IMAGE-'].update(data=get_pil_data(self.pil_cur_img), size=self.main_img_size)

                self.regen_thumbs()

            if event == '-FOLDER-':
                dir = sg.popup_get_folder('Open', no_window = True)
                print('OPENED:', dir)

                if dir:
                    files = [file for file in listdir(dir) if isfile(join(dir, file))]
                    if files.count == 0:
                        continue

                    self.cur_dir = dir
                    # Allocate history for each file we'll be filtering
                    #self.filt_hist = [(file, []) for file in files]
                    self.filt_hist = [FilterHistory(dir, file) for file in files]

                    self.cur_img_ind = 0
                    self.load_img()        
        window.close()

############################################################################

if __name__ == "__main__":
    GUI().main()
