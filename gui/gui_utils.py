import io
# PIL Utils

def get_pil_data(img):
    """Generate image data using PIL
    """
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def rescale(img, max_size):
    '''
    Rescale if too big to fit the max_size (MAIN_SIZE)
    '''
    largest_dim = max(img.size)
    largest_ind = 0 if img.size[0] == largest_dim else 1

    output = img

    if largest_dim > max_size[largest_ind]:
        scale = max_size[largest_ind] / largest_dim

        rescale_size = (int(img.size[0] * scale), int(img.size[1] * scale))

        output = img.resize(rescale_size)
    
    print("RESCALE: ", img.size, " --> ", output.size)
    return output