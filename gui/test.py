import PySimpleGUI as sg
from PIL import Image, ImageFilter, ImageOps
from io import BytesIO

main_img_path = './none.png'

FILTERS = [
    'A',
    'B',
    'C'
]

PREVIEWS = [
    './red.png', './green.png', './blue.png'
]

def gen_tile(event, i, img_path):
    return [
        [sg.Frame(event + str(i), layout=[
            [sg.Image(img_path, size=(200,200))],
            [sg.Button("Remove", key=f'-REMOVE-{i}')]
        ])]
    ]

main_col = sg.Column(
    [
        [sg.Button("Load", key='-LOAD-')],
        [sg.Image(main_img_path, key = '-IMAGE-', size=(500,500))],
        [
            # button for each frame
            sg.Frame(
                title=filter,
                layout=[[sg.Image(PREVIEWS[i], key = '-IMAGE-', size=(200,200))], [sg.Button(f'Filter {filter}', key = f'-FILTER-{filter}-')]],
                element_justification='c') for i, filter in enumerate(FILTERS)
        ]
    ], element_justification='c', key="-MAIN-", size=(800, 1000))

side_col = sg.Column([
    [sg.Frame("Original", layout=[[sg.Image(main_img_path, size=(200,200))]])]
    # [sg.Image(main_img_path, key = '-IMAGE-', size=(200,200))],
    # [sg.Button('AAA')]
    ], 
    scrollable=True, vertical_scroll_only=True, 
    # size_subsample_height=1, 
    size=(300, 600),
    justification='c',
    key="-SIDE-")

main_layout = [[main_col, side_col]]

# original = Image.open(image_path)
window = sg.Window('Image Editor', main_layout, resizable=True)
num_applied = 0

while True:
    event, values = window.read(timeout = 50)
    if event == sg.WIN_CLOSED:
        break
    
    #if event in ["-A-", "-B-", "-C-"]:
    if "FILTER" in event:
        window.extend_layout(window['-SIDE-'], gen_tile(event, num_applied, main_img_path))
            #[[sg.Frame(event + str(num_applied), layout=[[sg.Image(main_img_path, size=(200,200))]])]])
        window.refresh()
        window['-SIDE-'].contents_changed()

        num_applied += 1
    
    if event == '-LOAD-':
        new_path = sg.popup_get_file('Open',no_window = True)
        if new_path:
            main_img_path = new_path
        window['-IMAGE-'].update(filename=main_img_path, visible=True, size=(500,500))
        window.refresh()

    # if event == "-A-":
    #     # The refresh() and contents_changed() let it work with a scrollable container
    #     # If it is not scrollabe, don't need all that
    #     window.extend_layout(window['-SIDE-'], [[sg.Button("BBB", key="-BBB-")]])
    #     window.refresh()
    #     window['-SIDE-'].contents_changed()
    #     # print("extend")
    
    # update_image(
    #     original, 
    #     values['-BLUR-'],
    #     values['-CONTRAST-'], 
    #     values['-EMBOSS-'], 
    #     values['-CONTOUR-'],
    #     values['-FLIPX-'],
    #     values['-FLIPY-'])
    
        
window.close()