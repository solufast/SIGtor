import os
import cv2
import numpy as np
import random
from PIL import Image
from pathlib import Path


def get_file_paths(filepath, file_format=None, shuffle=False):
    """
    Get the file paths of files in a directory with a specific format.

    Args:
        filepath (str): The directory containing the files.
        file_format (list, optional): The file formats to include. If None, include all files.
        shuffle (bool, optional): If True, shuffle the file paths.

    Returns:
        list: The file paths.
    """
    if not os.path.isdir(filepath):
        raise ValueError(f"{filepath} is not a directory")

    if file_format is not None:
        file_paths = [os.path.join(filepath, filename) for filename in os.listdir(filepath)
                      if os.path.splitext(filename)[1].lower() in file_format]
    else:
        file_paths = [os.path.join(filepath, filename) for filename in os.listdir(filepath)]

    if shuffle:
        random.shuffle(file_paths)

    return file_paths


def read_and_resize_image(image_path, target_size):
    """Read an image from a file and resize it to the target size."""
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(image, target_size[:2])


def generate_random_image(target_size):
    """Generate a random RGB image of the target size."""
    image = np.ones(target_size)
    random_color = np.random.randint(0, 255, size=3).reshape(1, 3)
    image *= random_color
    return np.array(image, dtype=np.uint8)


def apply_gaussian_blur(image):
    """Apply Gaussian blur to an image."""
    return cv2.GaussianBlur(image, (3, 3), 0)


def get_backgrnd_image(bckgrnd_imgs_dir, target_img_size):
    """
    Get a background image.

    If there are suitable images in the specified directory, choose one at random,
    resize it to the target size, and apply Gaussian blur. If no suitable images are
    found, generate a random RGB image of the target size and apply Gaussian blur.

    Args:
        bckgrnd_imgs_dir (str): The directory containing the background images.
        target_img_size (tuple): The target size for the background image.

    Returns:
        numpy.ndarray: The background image.
    """
    if os.path.exists(bckgrnd_imgs_dir):
        image_paths = get_file_paths(bckgrnd_imgs_dir, file_format=['.jpg', '.png', '.jpeg'])
        if image_paths:
            image_path = random.choice(image_paths)
            image = read_and_resize_image(image_path, target_img_size)
        else:
            image = generate_random_image(target_img_size)
    else:
        image = generate_random_image(target_img_size)

    return apply_gaussian_blur(image)


def get_object_mask(source_image_path, cutout_coordinates, mask_directory, class_list=None):
    """
    Get the mask of an object in an image.

    Args:
        source_image_path (str): The path of the source image.
        cutout_coordinates (tuple): The coordinates of the object in the image.
        mask_directory (str): The directory of the mask images.
        class_list (list, optional): The list of classes. If None, all classes are included.

    Returns:
        Image: The mask of the object.
    """
    source_image_path = Path(source_image_path)
    mask_file_path = Path(mask_directory) / f"{source_image_path.stem}.png"

    if not source_image_path.exists():
        raise ValueError(f"{source_image_path} does not exist")
    if not mask_file_path.exists():
        raise ValueError(f"{mask_file_path} does not exist")

    try:
        mask_image = Image.open(mask_file_path)
        object_mask = mask_image.crop(box=cutout_coordinates)
    except IOError:
        x1, y1, x2, y2 = cutout_coordinates
        object_mask = Image.fromarray(
            255 * np.ones((y2 - y1, x2 - x1), dtype=np.uint8)
        )

    return object_mask


def get_outer_box(original_boxes, image_size=(None, None), add_random_offset=False, min_offset=1, max_offset=5):
    """
    Get the outer box coordinates based on the original boxes.

    Args:
        original_boxes (np.array): The original bounding box coordinates.
        image_size (tuple): The dimensions of the image.
        add_random_offset (bool): Whether to add a random offset to the box coordinates.
        min_offset (int): The minimum random offset.
        max_offset (int): The maximum random offset.

    Returns:
        np.array: The outer box coordinates.
    """
    if original_boxes.size == 0 or original_boxes.shape[1] != 4:
        raise ValueError("Invalid original boxes")

    x_min, y_min = np.min(original_boxes[..., :2], axis=0)
    x_max, y_max = np.max(original_boxes[..., 2:], axis=0)

    if all(image_size) and add_random_offset:
        width, height = image_size
        x_min = max(x_min - random.randint(min_offset, max_offset), 0)
        y_min = max(y_min - random.randint(min_offset, max_offset), 0)
        x_max = min(x_max + random.randint(min_offset, max_offset), width)
        y_max = min(y_max + random.randint(min_offset, max_offset), height)

    return np.array([x_min, y_min, x_max, y_max]).reshape(-1, 4)


def get_data(annotation_line, mask_dir):
    """
    Get data from an annotation line.

    Args:
        annotation_line (str): The annotation line.
        mask_dir (str): The path to the directory containing mask images.

    Returns:
        tuple: A tuple containing the image path, object image, object mask, outer box, and inner boxes.
    """
    if not annotation_line:
        raise ValueError("Invalid annotation line")

    split_line = annotation_line.split()
    image_path = split_line[0]

    try:
        original_image = Image.open(image_path)
    except IOError:
        print(f"Unable to open image at {image_path}")
        return

    original_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in split_line[1:]]).reshape(-1, 5)
    outer_box = get_outer_box(original_boxes, original_image.size, add_random_offset=False, min_offset=0, max_offset=15)
    inner_boxes = original_boxes

    object_classes = list(inner_boxes[:, 4]) if len(inner_boxes[:, 4]) else None
    object_image = original_image.crop(box=outer_box[0, 0:4])
    object_mask = get_object_mask(image_path, outer_box[0, 0:4], mask_dir, class_list=object_classes)

    return image_path, object_image, object_mask, outer_box, inner_boxes


def random_augmentations(obj_img, obj_mask, outerbox, inner_boxes, max_augs=2, random_aug_nums=False):
    """
    Apply random augmentations to the object image and object mask.

    Args:
        obj_img (Image): The object image.
        obj_mask (Image): The object mask.
        outerbox (ndarray): The outer box coordinates.
        inner_boxes (ndarray): The inner box coordinates.
        max_augs (int, optional): The maximum number of augmentations. Defaults to 2.
        random_aug_nums (bool, optional): Whether the number of augmentations should be random. Defaults to False.

    Returns:
        tuple: The augmented object image, object mask, outer box coordinates, and inner box coordinates.
    """

    # Define the augmentation functions
    augmentation_functions = {
        'blur': random_blur,
        'grayscale': random_grayscale,
        'brightness': random_brightness,
        'sharpness': random_sharpness,
        'chroma': random_chroma,
        'contrast': random_contrast,
        'hsv': random_hsv_distort,
        'vflip': flip_vertical,
        'hflip': flip_horizontal,
        'rescale': rescale
    }

    augmentation_types = list(augmentation_functions.keys())

    if random_aug_nums:
        max_augs = np.random.randint(0, len(augmentation_types))

    max_augs = min(max_augs, len(augmentation_types))

    selected_augmentations = np.random.choice(augmentation_types, size=max_augs, replace=False)

    # Ensure 'rescale' is always included
    if 'rescale' not in selected_augmentations:
        selected_augmentations.append('rescale')

    for aug_type in selected_augmentations:
        function = augmentation_functions[aug_type]
        obj_img, obj_mask, outerbox, inner_boxes = function(obj_img, obj_mask, outerbox, inner_boxes)

    return obj_img, obj_mask, outerbox, inner_boxes
