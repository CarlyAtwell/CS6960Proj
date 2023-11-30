from PIL import Image, ImageEnhance


'''
Enhances brightness of the img

img: PIL image
val: percentage; 1.0 = no change, 0.5 = decrease by 50%, 1.5 = increase by 50%
'''
def brightness(img, val):
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