import os.path
from PIL import Image


def get_img_dir():
    src_dir = os.path.dirname(__file__)
    img_dir = os.path.join(src_dir, '../../..', 'img')
    return img_dir


def open_img():
    img_dir = get_img_dir()
    img_name = 'single_user.jpg'
    full_img_path = os.path.join(img_dir, img_name)
    img = Image.open(full_img_path)
    return img


def crop_image(img, xy, scale_factor):
    '''Crop the image around the tuple xy

    Inputs:
    -------
    img: Image opened with PIL.Image
    xy: tuple with relative (x,y) position of the center of the cropped image
        x and y shall be between 0 and 1
    scale_factor: the ratio between the original image's size and the cropped image's size
    '''
    center = (img.size[0] * xy[0], img.size[1] * xy[1])
    new_size = (img.size[0] / scale_factor, img.size[1] / scale_factor)
    left = max (0, (int) (center[0] - new_size[0] / 2))
    right = min (img.size[0], (int) (center[0] + new_size[0] / 2))
    upper = max (0, (int) (center[1] - new_size[1] / 2))
    lower = min (img.size[1], (int) (center[1] + new_size[1] / 2))
    cropped_img = img.crop((left, upper, right, lower))
    return cropped_img


def save_img(img, img_name):
    img_dir = get_img_dir()
    full_img_path = os.path.join(img_dir, img_name)
    img.save(full_img_path)


if __name__ == '__main__':
    ams = open_img()

    crop_ams = crop_image(ams, (0.50, 0.50), 0.95)
    save_img(crop_ams, 'crop_amsterdam_01.jpg')

    crop_ams = crop_image(ams, (0.25, 0.25), 2.5)
    save_img(crop_ams, 'crop_amsterdam_02.jpg')

    crop_ams = crop_image(ams, (0.75, 0.45), 3.5)
    save_img(crop_ams, 'crop_amsterdam_03.jpg')