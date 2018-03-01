"""
Description: Clean exif data for all images in given directory because
keras can't read some of them.
"""
import os
from PIL import Image

WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORKING_DIR, '..', 'data', '144k_flow')

def clean_exif_for_image(img_path, new_img_path):
    """ Clean exif data for one image """
    image_file = open(img_path, 'rb')
    image = Image.open(image_file)

    # next 3 lines strip exif
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)

    image_without_exif.save(new_img_path)


def clean_exif_in_folder(dir_name, new_dir=None):
    """ Clean exif data in a given folder """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        raise ValueError("{} should be a directory".format(dir_name))
    if new_dir is None:
        new_dir = dir_name + '_exif'
    if os.path.exists(new_dir):
        raise ValueError("Directory {} already exists".format(new_dir))
    else:
        os.makedirs(new_dir)
    for file_name in os.listdir(dir_name):
        full_path = os.path.join(dir_name, file_name)
        full_path_new = os.path.join(new_dir, file_name)
        if os.path.isdir(full_path):
            clean_exif_in_folder(full_path, full_path_new)
        else:
            clean_exif_for_image(full_path, full_path_new)


def execute():
    """ Execute script """
    clean_exif_in_folder(DATA_DIR)


if __name__ == "__main__":
    execute()