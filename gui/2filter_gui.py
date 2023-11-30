import PySimpleGUI as sg
import cv2 as cv
import numpy as np

# Modules
import utils
import filters

### UTILS ##################################################################
def get_img_bytes(img):
    return cv.imencode('.ppm', img)[1].tobytes() # can also use png
############################################################################

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
############################################################################

#### FILTERS ###############################################################
def sepia(img):
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia

# # brightness adjustment
# def bright(img, beta_value ):
#     img_bright = cv.convertScaleAbs(img, beta=beta_value)
#     return img_bright
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



def gen_filter_group(group):
    name = group[0]
    filts = group[1]

    tiles = []

    for key in filts:
        tiles.append(sg.Column(
            [[sg.Image(default_path, key=f"-FILTER-{key}", size=THUMBNAIL_SIZE), sg.Text(key)]],
            element_justification='c'))

    return sg.Frame(title=name, layout=[tiles])




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

# main_col = sg.Column(
#     [
#         [sg.Column([
#             [sg.Button("Load", key='-LOAD-'), sg.Button("Folder", key='-FOLDER-')]
#             ], justification='l')],
        
#         [sg.Image(main_img_path, key = '-IMAGE-', size=(500,500))],
#         [
#             # button for each filter
#             sg.Frame(
#                 title=filter,
#                 layout=[[sg.Image(PREVIEWS[i], key = f'-IMAGE-{i}', size=(200,200))], [sg.Button(f'Filter {filter}', key = f'-FILTER-{filter}-')]],
#                 element_justification='c') for i, filter in enumerate(FILTERS)
#         ]
#     ], 
#     element_justification='c', 
#     key="-MAIN-", 
#     size=SIZES['-MAIN-'])

# side_col = sg.Column([], 
#     scrollable=True, vertical_scroll_only=True, 
#     # size_subsample_height=1, 
#     size=SIZES['-SIDE-'],
#     justification='c',
#     key="-SIDE-")

main_layout = [[main_col, filter_col]]

window = sg.Window('Image Editor', main_layout, resizable=True)
num_applied = 0

def main():
    curr_win_size = (0,0)
    curr_img = cv.imread(default_path) if default_path else None

    while True:
        event, values = window.read(timeout = 50)
        if event == sg.WIN_CLOSED:
            break

        if not utils.vec_equal2d(curr_win_size, window.size):
            curr_win_size = window.size
            print(curr_win_size)
            #TODO: change heights of elements, refresh
            # window['-SIDE-'].set_size((100, 100))
            set_size(window['-SIDE-'], (SIZES['-SIDE-'][0], curr_win_size[1]-WIN_PAD))
            # window.refresh()
            # window['-SIDE-'].contents_changed()
        
        #if event in ["-A-", "-B-", "-C-"]:
        if "FILTER" in event:
            window.extend_layout(window['-SIDE-'], gen_tile(event, num_applied, curr_img))
            window['-SIDE-']
                #[[sg.Frame(event + str(num_applied), layout=[[sg.Image(main_img_path, size=(200,200))]])]])
            window.refresh()
            window['-SIDE-'].contents_changed()

            num_applied += 1
        
        if event == '-LOAD-':
            new_path = sg.popup_get_file('Open',no_window = True)
            if new_path:
                main_img_path = new_path
                img = cv.imread(main_img_path)
                curr_img = img

                img_bytes = get_img_bytes(img)
                
                window['-IMAGE-'].update(data=img_bytes, visible=True, size=(500,500))
                window.refresh()

        if event == '-FOLDER-':
            f = sg.popup_get_folder('Open', no_window = True)
            print(f)
            # How to display opencv result
            img = sepia(cv.imread('./demo.jpg'))
            img_bytes = cv.imencode('.ppm', img)[1].tobytes() # can also use png
            
            window['-IMAGE-'].update(data=img_bytes)

            
    window.close()

############################################################################

if __name__ == "__main__":
    main()
