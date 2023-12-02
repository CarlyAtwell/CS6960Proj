from PIL import Image, ImageEnhance

def brightness(img, val):
    '''
    Enhances brightness of the img

    img: PIL image
    val: percentage; 1.0 = no change, 0.5 = decrease by 50%, 1.5 = increase by 50%
    '''
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(val)

def color_balance(img, val):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(val)

def contrast(img, val):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(val)

def sharpness(img, val):
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(val)


# (name, func_ptr, args_dict)
FILTERS = {
    'bright+10': (brightness, {'val': 1.1}),
    'bright-10': (brightness, {'val': 0.9}),
    'color_bal+': (color_balance, {'val': 2.0}),
    'color_bal-': (color_balance, {'val': 0.5}),
    'contrast+10': (contrast, {'val': 1.1}),
    'contrast-10': (contrast, {'val': 0.9}),
    'sharpness+': (sharpness, {'val': 2.0}),
    'sharpness-': (sharpness, {'val': 0.0})
    #TODO: channel mixing
}


def apply_filter_chain(img, filters):
    '''
    Takes in Pillow image and applies a list of filters in order
    '''
    filtered = img.copy()
    for filter in filters:
        filtered = apply_filter(filtered)

    return filtered

def apply_filter(img, filter):
    '''
    Takes in Pillow image and applies a filter
    '''

    func, args = FILTERS[filter]
    return func(img, **args)