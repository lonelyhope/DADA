import os
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def _all_images(path, sort=True):
    """
    return all images in the folder
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    abs_path = os.path.abspath(path)
    image_files = list()
    for subpath in os.listdir(abs_path):
        if os.path.isdir(os.path.join(abs_path, subpath)):
            image_files = image_files + _all_images(os.path.join(abs_path, subpath))
        else:
            if _is_image_file(subpath):
                image_files.append(os.path.join(abs_path, subpath))
    if sort:
        image_files.sort()

    if '/data2/xxq/SR/REDS/reds/train/train_sharp' in path:
        image_files2 = []
        for image_file in image_files:
            name = image_file.split('/')[-1]
            # print(name)
            if name == '00000024.png':
                # print('add', name)
                image_files2.append(image_file)
        return image_files2

    return image_files

def pil_loader(path, mode='RGB'):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: PIL.Image
    """
    assert _is_image_file(path), "%s is not an image" % path
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)