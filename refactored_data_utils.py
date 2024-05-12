import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def letterbox_resize(image, target_size, return_padding_info=False):
    """Resize image, keeping aspect ratio with optional padding info."""
    src_w, src_h = image.size
    target_w, target_h = target_size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    image = image.resize((new_w, new_h), Image.ANTIALIAS)

    if return_padding_info:
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        padding_size = (target_w - new_w, target_h - new_h)
        offset = (dx, dy)
        return image, padding_size, offset
    return image


def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Enhance brightness, contrast, and sharpness of an image."""
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image


def apply_random_transformation(image, max_rotation=0, max_translation=(0, 0), max_scale=1.0):
    """Apply random rotation, translation, and scaling to an image."""
    angle = random.uniform(-max_rotation, max_rotation)
    tx, ty = random.uniform(-max_translation[0], max_translation[0]), random.uniform(-max_translation[1],
                                                                                     max_translation[1])
    scale = random.uniform(1.0, max_scale)
    image = image.rotate(angle, expand=True)
    image = image.transform(image.size, Image.AFFINE, (scale, 0, tx, 0, scale, ty))
    return image


def apply_filter(image, filter_type=None):
    """Apply predefined filter to an image."""
    if filter_type == 'BLUR':
        return image.filter(ImageFilter.BLUR)
    elif filter_type == 'CONTOUR':
        return image.filter(ImageFilter.CONTOUR)
    elif filter_type == 'EDGE_ENHANCE':
        return image.filter(ImageFilter.EDGE_ENHANCE)
    return image


# Example usage of the functions within a script or application context
if __name__ == '__main__':
    # Load an image using PIL
    img_path = './Datasets/Source/Images/2010_001024.jpg'
    img = Image.open(img_path)

    # Resize the image
    resized_img = letterbox_resize(img, (640, 480))

    # Enhance the image
    enhanced_img = enhance_image(resized_img, brightness=1.2, contrast=1.3)

    # Apply a random transformation
    transformed_img = apply_random_transformation(enhanced_img, max_rotation=10, max_translation=(15, 15),
                                                  max_scale=1.2)

    # Apply a filter
    filtered_img = apply_filter(transformed_img, 'BLUR')

    filtered_img.save("misc/images/2009_004125_processed.jpg")
