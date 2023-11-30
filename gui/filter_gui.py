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

#### GUI HELPERS ###########################################################
def set_size(element, size):
    # Only work for sg.Column when `scrollable=True` or `size not (None, None)`
    options = {'width':size[0], 'height':size[1]}
    if element.Scrollable or element.Size!=(None, None):
        element.Widget.canvas.configure(**options)
    else:
        element.Widget.pack_propagate(0)
        element.set_size(size)

# def gen_tile(event, i, curr_img):
#     img_bytes = get_img_bytes(curr_img)
#     return [
#         [sg.Frame(event + str(i), layout=[
#             [sg.Image(data=img_bytes, size=(100,100))],
#             [sg.Button("Remove", key=f'-REMOVE-{i}')]
#         ])]
#     ]

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

DEFAULT_DIR = './default'

# GUI CONSTANTS ##########
SIZES = {
    '-MAIN-': (800, 1000),
    '-SIDE-': (200, 600)
}

WIN_PAD = 75

THUMBNAIL_SIZE = (200,200)
MAIN_SIZE = (800,800)
##########################

# # (name, func_ptr, args_dict)
# FILTERS = {
#     'bright+10': (filters.brightness, {'val': 1.1}),
#     'bright-10': (filters.brightness, {'val': 0.9}),
#     'color_bal+': (filters.color_balance, {'val': 2.0}),
#     'color_bal-': (filters.color_balance, {'val': 0.5}),
#     'contrast+10': (filters.contrast, {'val': 1.1}),
#     'contrast-10': (filters.contrast, {'val': 0.9}),
#     'sharpness+': (filters.sharpness, {'val': 2.0}),
#     'sharpness-': (filters.sharpness, {'val': 0.0})
#     #TODO: channel mixing
# }

FILTER_GROUPS = [
    ('Brightness', ['bright-10', 'bright+10']),
    ('Color Balance', ['color_bal-', 'color_bal+']),
    ('Contrast', ['contrast-10', 'contrast+10']),
    ('Sharpness', ['sharpness-', 'sharpness+']),
]

MAX_FILTER_APPLICATIONS = 5
############################################################################

main_col = sg.Column(
    [
        [sg.Column([
            [sg.Button("Load Folder", key='-FOLDER-'), sg.Button("Export Trajectories", key='-EXPORT-')]
            ], justification='l')],
        
        [sg.Image(default_path, key = '-MAIN_IMAGE-', size=MAIN_SIZE)],
        [sg.Text("No Filters Applied", key="-LABEL-")]
    ], 
    element_justification='c', 
    key="-MAIN-", 
    size=SIZES['-MAIN-'])

next_btns = [[sg.Button("<", key="-PREV-"), sg.Text("0/0", key="-LABEL_NUM_IMG-"), sg.Button(">", key="-NEXT-"), 
              sg.Push(), 
              sg.Text(f"Thumb Size: {THUMBNAIL_SIZE}", key='-LABEL_THUMBSIZE-'), sg.Button("-", key='-THUMB_M-'), sg.Button("+", key='-THUMB_P-')]]
hist_btns = [[sg.Button("Undo", key="-UNDO-"), sg.Button("Redo", key="-REDO-"), sg.Text(f"0/{MAX_FILTER_APPLICATIONS}", key="-LABEL_NUM_APPS-")]]
filt_btns = [[gen_filter_group(group)] for group in FILTER_GROUPS]
filter_col = sg.Column((next_btns + hist_btns + filt_btns), scrollable=True, vertical_scroll_only=True, size_subsample_height=1)

#TODO: figure out how to place two cols next to each other properly for layout to work
# Two col layout
# filt_col_alt1 = sg.Column(filt_btns[:len(filt_btns)//2])
# filt_col_alt2 = sg.Column(filt_btns[len(filt_btns)//2:])
# filt_col_alt = sg.Column([[filt_col_alt1, sg.HSeparator(), filt_col_alt2]])
# hist_col = sg.Column([],
#     scrollable=True, vertical_scroll_only=True, 
#     # size_subsample_height=1, 
#     size=SIZES['-SIDE-'],
#     justification='c',
#     key="-HIST-")

# main_layout = [[main_col, filter_col, hist_col]]
main_layout = [[main_col, filter_col]]

window = sg.Window('Image Editor', main_layout, resizable=True)

# PIL
def get_pil_data(img):
    """Generate image data using PIL
    """
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def rescale(img):
    '''
    Rescale if too big to fit
    '''
    largest_dim = max(img.size)
    largest_ind = 0 if img.size[0] == largest_dim else 1

    output = img

    if largest_dim > MAIN_SIZE[largest_ind]:
        scale = MAIN_SIZE[largest_ind] / largest_dim

        rescale_size = (int(img.size[0] * scale), int(img.size[1] * scale))

        output = img.resize(rescale_size)
    
    print("RESCALE: ", img.size, " --> ", output.size)
    return output

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

        # self.load()

    def __repr__(self) -> str:
        # return '{' + self.filename + ', ' + str(self.get_filts()) + '}'
        #return self.filename + ' ' + str(self.get_filts())
        return self.filename + ' ' + 'out_' + self.filename + ' ' + str(self.get_filts())

    def clean(self):
        '''
        To preserve space

        Purges all image data

        Use load() to reload data
        '''
        #TODO: I think don't bother with this maybe so don't have to do load() again, might make things a little faster
        self.images = []
        self.base_img = None

    def get_cur(self):
        '''
        Gets the current image based on index position
        '''
        if self.ind == -1:
            return self.base_img
        return self.images[self.ind]
    
    def get_filts(self):
        '''
        Returns list of filters applied to the index
        '''

        return self.filters[:self.ind+1]
    
    def get_num_active(self):
        return self.ind + 1

    def load(self, do_rescale=True):
       '''
       Reloads images from the current filter pool
       '''
       self.base_img = rescale(Image.open(self.filepath)) if do_rescale else Image.open(self.filepath)

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
        self.filters = self.filters[:self.ind+1]
        self.images = self.images[:self.ind+1]

        cur_img = self.images[self.ind] if self.ind != -1 else self.base_img

        func, args = FILTERS[filter]
        cur_img = func(cur_img, **args)
        self.images.append(cur_img)
        self.filters.append(filter)
        self.ind += 1

        # print(self.filters)

    def set_active(self, ind):
        '''
        Change which of the history is active
        '''
        self.ind = ind

    def undo(self):
        self.ind -= 1
        if self.ind < -1:
            self.ind = -1
            return False
        return True

    def redo(self):
        self.ind += 1
        if self.ind > (len(self.filters) - 1):
            self.ind = len(self.filters) - 1
            return False
        return True

class GUI:
    thumb_size = THUMBNAIL_SIZE # To allow rescaling of thumbnails if screen is too small

    cur_dir = None
    num_dir_images = 0

    cur_img_ind = -1
    pil_cur_img = None
    pil_thumbnail = None
    filt_hist = []

    main_img_size = MAIN_SIZE
    curr_win_size = (0,0)

    def load_img(self):
        cur_hist = self.filt_hist[self.cur_img_ind]
        self.pil_cur_img = cur_hist.get_cur()

        # Set filter text
        window['-LABEL-'].update('Applied: ' + str(cur_hist.get_filts()))

        # Set preview
        window['-MAIN_IMAGE-'].update(data=get_pil_data(self.pil_cur_img), size=self.main_img_size)
        #window['-MAIN_IMAGE-'].update(data=get_pil_data(self.pil_cur_img))

        # Regen thumbs
        self.regen_thumbs()

        # Update applications label
        window['-LABEL_NUM_APPS-'].update(f'{cur_hist.get_num_active()}/{MAX_FILTER_APPLICATIONS}')

    def regen_thumbs(self):
        #TODO: this might be making performance bad; maybe b/c now have undo/redo just get rid of the previews? but they are nice
        
        # Apply all filters to thumnail and update previews
        self.pil_thumbnail = self.pil_cur_img.resize(self.thumb_size)
        for key, value in FILTERS.items():
            func, args = value

            thumb_filtered = func(self.pil_thumbnail, **args)

            window[f'-THUMB-{key}'].update(image_data=get_pil_data(thumb_filtered), visible=True)

    def redraw_hist(self):
        #TODO: if end up drawing all history in side panel
        pass

    def load_folder(self, dir):
        files = [file for file in listdir(dir) if isfile(join(dir, file)) and file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        if files.count == 0:
            return

        self.cur_dir = dir
        self.num_dir_images = len(files)
        print(f"Found {self.num_dir_images} images ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')")
        # Allocate history for each file we'll be filtering
        self.filt_hist = [FilterHistory(dir, file) for file in files]

        self.cur_img_ind = 0
        self.filt_hist[self.cur_img_ind].load()
        self.load_img()

        # Change label
        window['-LABEL_NUM_IMG-'].update(f'{self.cur_img_ind+1}/{self.num_dir_images}')

    def main(self):
        init = False

        while True:
            event, values = window.read(timeout = 50)
            if event == sg.WIN_CLOSED:
                break

            if not init:
                self.load_folder(DEFAULT_DIR)
                init = True

            # # TODO: here can handle resize events
            # if not utils.vec_equal2d(self.curr_win_size, window.size):
            #     self.curr_win_size = window.size
            #     # set_size(window['-HIST-'], (SIZES['-SIDE-'][0], self.curr_win_size[1]-WIN_PAD))

            if event == '-UNDO-':
                if self.filt_hist[self.cur_img_ind].undo():
                    self.load_img()

            if event == '-REDO-':
                if self.filt_hist[self.cur_img_ind].redo():
                    self.load_img()

            if event == '-PREV-':
                if self.cur_img_ind > 0:
                    self.filt_hist[self.cur_img_ind].clean()
                    
                    self.cur_img_ind -= 1
                    # Reload cleaned data in history
                    self.filt_hist[self.cur_img_ind].load()
                    # Load image
                    self.load_img()

                    # Change label
                    window['-LABEL_NUM_IMG-'].update(f'{self.cur_img_ind+1}/{self.num_dir_images}')
            
            if event == '-NEXT-':
                if self.cur_img_ind < self.num_dir_images - 1:
                    self.filt_hist[self.cur_img_ind].clean()
                    
                    self.cur_img_ind += 1
                    # Reload cleaned data in history
                    self.filt_hist[self.cur_img_ind].load()
                    # Load image
                    self.load_img()

                    # Change label
                    window['-LABEL_NUM_IMG-'].update(f'{self.cur_img_ind+1}/{self.num_dir_images}')

            if event == '-THUMB_M-':
                # Don't resize if too small
                if self.thumb_size[0] > 20:
                    self.thumb_size = (self.thumb_size[0] - 20, self.thumb_size[1] - 20)
                    self.regen_thumbs()

                    # Set label
                    window['-LABEL_THUMBSIZE-'].update(f'Thumb Size: {self.thumb_size}')

            if event == '-THUMB_P-':
                # Don't resize if too large
                if self.thumb_size[0] < 400:
                    self.thumb_size = (self.thumb_size[0] + 20, self.thumb_size[1] + 20)
                    self.regen_thumbs()

                    # Set label
                    window['-LABEL_THUMBSIZE-'].update(f'Thumb Size: {self.thumb_size}')
            
            if "-THUMB-" in event:
                # print("FILTER:", event)

                # Retrieve filter key
                key = event[7:]
                
                # Apply filter, update history
                cur_hist = self.filt_hist[self.cur_img_ind]

                # Don't apply if more than max allowed
                if cur_hist.get_num_active() < MAX_FILTER_APPLICATIONS:
                    cur_hist.apply(key)

                    self.load_img()

            if event == '-FOLDER-':
                dir = sg.popup_get_folder('Open', no_window = True)
                print('OPENED:', dir)

                if dir:
                    self.load_folder(dir)
                    # files = [file for file in listdir(dir) if isfile(join(dir, file)) and file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
                    # if files.count == 0:
                    #     continue

                    # self.cur_dir = dir
                    # self.num_dir_images = len(files)
                    # print(f"Found {self.num_dir_images} images ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')")
                    # # Allocate history for each file we'll be filtering
                    # self.filt_hist = [FilterHistory(dir, file) for file in files]

                    # self.cur_img_ind = 0
                    # self.filt_hist[self.cur_img_ind].load()
                    # self.load_img()

                    # # Change label
                    # window['-LABEL_NUM_IMG-'].update(f'{self.cur_img_ind+1}/{self.num_dir_images}')
            if event == '-EXPORT-':
                # Only if exists
                if self.cur_dir:

                    print('\n\nEXPORT: ', self.filt_hist)
                    print('\n\n')

                    export_name = f'export_{time.time()}'
                    path = f'{self.cur_dir}/{export_name}'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    # TODO make this better, json, or convert it to a numpy array and do the numpy.savetxt so we can then load it easily in the other one
                    with open(f'{path}/{export_name}.txt', "w") as f:
                        # f.write(str(self.filt_hist))
                        f.write(str(len(self.filt_hist)))
                        f.write('\n')
                        for hist in self.filt_hist:
                            f.write(str(hist))
                            f.write('\n')
                    
                    #np.savetxt(f'{self.cur_dir}/export.npy', np.array(self.filt_hist))

                    # Save output images for this export
                    for hist in self.filt_hist:
                        # We use rescale inside hist normally, so now don't do that
                        hist.load(do_rescale=False)
                        pil_img = hist.get_cur()
                        pil_img.save(f'{path}/out_{hist.filename}')
                        # Now unload
                        hist.clean()

                    # Reload the current one
                    self.filt_hist[self.cur_img_ind].load()

        window.close()

############################################################################

if __name__ == "__main__":
    GUI().main()
