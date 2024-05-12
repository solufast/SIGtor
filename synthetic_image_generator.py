import argparse
import copy
import math
import os
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_utils import random_blur, random_grayscale, random_brightness, random_sharpness, \
    random_chroma, random_contrast, random_hsv_distort, random_vertical_flip, random_horizontal_flip, \
    random_size
from utils import read_ann, get_file_paths, rand, overlap_measure, convert_to_ann_line, get_colors, \
    parse_commandline_arguments


def get_next_source(current_indx, total_indexs, randomly=False):
    if randomly:
        return np.random.randint(0, total_indexs - 1)
    if current_indx < total_indexs - 1:
        return current_indx + 1
    else:
        return 0


def get_backgrnd_image(bckgrnd_imgs_dir, target_img_size):
    if os.path.exists(bckgrnd_imgs_dir):
        bckgrnd_imgs_list = get_file_paths(bckgrnd_imgs_dir, file_format=['.jpg', '.png', '.jpeg'])
    else:
        bckgrnd_imgs_list = []
    if len(bckgrnd_imgs_list) != 0:
        np.random.shuffle(bckgrnd_imgs_list)
        bcgrnd_img_path = np.random.choice(bckgrnd_imgs_list)
        bcgrnd_img = cv2.cvtColor(cv2.imread(bcgrnd_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        bcgrnd_img = cv2.resize(bcgrnd_img, target_img_size[:2])
    else:
        bcgrnd_img = np.ones(target_img_size)
        random_color = np.random.randint(0, 255, size=3).reshape(1, 3)
        bcgrnd_img *= random_color
        bcgrnd_img = np.array(bcgrnd_img).astype(np.uint8)
    bcgrnd_img = cv2.GaussianBlur(bcgrnd_img, (3, 3), 0)
    return bcgrnd_img


def get_objmask(src_imgpath, cutout_coord, maskdir, class_list=None):
    fname = os.path.splitext(os.path.basename(src_imgpath))[0]
    mfname = os.path.join(maskdir, fname + ".png")
    if os.path.exists(mfname):
        mask_img = Image.open(mfname)
        obj_mask = mask_img.crop(box=cutout_coord)
        # obj_mask_array = np.array(obj_mask)
        # total_objs_in_cutout = len(np.unique(obj_mask_array))
        # obj_mask_array[obj_mask_array == 255] = total_objs_in_cutout  # Replace white boarder around the objects
        # obj_mask = Image.fromarray(obj_mask_array)
    else:
        x1, y1, x2, y2 = cutout_coord
        outerbox_width = (x2 - x1)
        outerbox_height = (y2 - y1)
        obj_mask = Image.fromarray(255 * np.ones((outerbox_height, outerbox_width), dtype=np.uint8))
    return obj_mask


def get_outerbox(org_boxes, img_size=(None, None), addrandomoffset=False, minoffset=1, maxoffset=5):
    if (img_size[0] is not None and img_size[1] is not None) and addrandomoffset:
        w, h = img_size
        x1 = max(np.min(org_boxes[..., 0]) - int(random.randint(minoffset, maxoffset)), 0)
        y1 = max(np.min(org_boxes[..., 1]) - int(random.randint(minoffset, maxoffset)), 0)
        x2 = min(np.max(org_boxes[..., 2]) + int(random.randint(minoffset, maxoffset)), w)
        y2 = min(np.max(org_boxes[..., 3]) + int(random.randint(minoffset, maxoffset)), h)
        outerbox = np.array([x1, y1, x2, y2]).reshape(-1, 4)
    else:
        x1 = np.min(org_boxes[..., 0])
        y1 = np.min(org_boxes[..., 1])
        x2 = np.max(org_boxes[..., 2])
        y2 = np.max(org_boxes[..., 3])
        outerbox = np.array([x1, y1, x2, y2]).reshape(-1, 4)
    return outerbox


def get_data(annotation_line, maskdir):
    line = annotation_line.split()
    imgpath = line[0]
    org_img = Image.open(imgpath)
    org_boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]]).astype('int32').reshape(-1,
                                                                                                                   5)
    outerbox = get_outerbox(org_boxes, org_img.size, addrandomoffset=False, minoffset=0, maxoffset=15)
    inner_boxes = org_boxes
    obj_classes = list(inner_boxes[:, 4]) if len(inner_boxes[:, 4]) != 0 else None
    obj_img = org_img.crop(box=outerbox[0, 0:4])
    obj_mask = get_objmask(imgpath, outerbox[0, 0:4], maskdir, class_list=obj_classes)
    return imgpath, obj_img, obj_mask, outerbox, inner_boxes


def random_augmentations(obj_img, obj_mask, outerbox, inner_boxes, max_augs=2, random_aug_nums=False):
    augtypes = ['rescale', 'blur', 'brightness', 'chroma', 'contrast', 'hsv', 'hflip', 'sharpness', 'grayscale',
                'vflip']

    if random_aug_nums:
        max_augs = np.random.randint(0, len(augtypes))
    max_augs = max_augs if max_augs <= len(augtypes) else len(augtypes)
    selected_augtypes = list(np.random.choice(augtypes, size=max_augs, replace=False))

    if 'rescale' not in selected_augtypes:
        selected_augtypes.append('rescale')  # and let us make 'rescale' a mandatory augmentation

    new_obj_img = obj_img
    new_obj_mask = obj_mask
    new_outerbox = outerbox
    new_innerboxes = inner_boxes

    for augtpe in selected_augtypes:
        if augtpe == 'blur':
            new_obj_img = random_blur(new_obj_img)
        if augtpe == 'grayscale':
            new_obj_img = random_grayscale(new_obj_img)
        if augtpe == 'brightness':
            new_obj_img = random_brightness(new_obj_img)
        if augtpe == 'sharpness':
            new_obj_img = random_sharpness(new_obj_img)
        if augtpe == 'chroma':
            new_obj_img = random_chroma(new_obj_img)
        if augtpe == 'contrast':
            new_obj_img = random_contrast(new_obj_img)
        if augtpe == 'hsv':
            new_obj_img = random_hsv_distort(new_obj_img)
        if augtpe == 'vflip':
            new_obj_img, _ = random_vertical_flip(new_obj_img, prob=1.0)
            new_obj_mask, _ = random_vertical_flip(new_obj_mask, prob=1.0)
            x1, y1, x2, y2 = new_outerbox[0, 0:4]
            h = (y2 - y1)
            new_outerbox = np.array([x1, y1, x2, y2, -1]).reshape(-1, 5)  # -1 is placeholder holder for object class
            # as the outer box does not a class
            boxes = np.concatenate((new_outerbox, new_innerboxes), axis=0)
            boxes[..., [1, 3]] = h - boxes[..., [3, 1]]
            # extract the new rescaled outer and inner boxes
            new_outerbox = np.array(boxes[0, 0:4]).reshape(-1, 4)
            new_innerboxes = np.array(boxes[1:, 0:5]).reshape(-1, 5)
        if augtpe == 'hflip':
            new_obj_img, _ = random_horizontal_flip(new_obj_img, prob=1.0)
            new_obj_mask, _ = random_horizontal_flip(new_obj_mask, prob=1.0)
            x1, y1, x2, y2 = new_outerbox[0, 0:4]
            w = (x2 - x1)
            new_outerbox = np.array([x1, y1, x2, y2, -1]).reshape(-1, 5)
            boxes = np.concatenate((new_outerbox, new_innerboxes), axis=0)
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
            # extract the new rescaled outer and inner boxes
            new_outerbox = np.array(boxes[0, 0:4]).reshape(-1, 4)
            new_innerboxes = np.array(boxes[1:, 0:5]).reshape(-1, 5)
        if augtpe == 'rescale':
            x1, y1, x2, y2 = new_outerbox[0, 0:4]
            new_outerbox = np.array([x1, y1, x2, y2, -1]).reshape(-1, 5)
            boxes = np.concatenate((new_outerbox, new_innerboxes), axis=0)
            if ((y2 - y1) * (x2 - x1)) < 32 ** 2:
                scale = rand(1.5, 3.0)
            elif ((y2 - y1) * (x2 - x1)) < 96 ** 2:
                scale = rand(1.0, 2.0)
            else:
                scale = rand(0.9, 1.5)
            scale_w_or_h = rand()
            new_obj_img, new_obj_mask, boxes = random_size(new_obj_img, mask=new_obj_mask, boxes=boxes, scale=scale,
                                                           prob=scale_w_or_h, keepaspectratio=True)
            new_outerbox = np.array(boxes[0, 0:4]).reshape(-1, 4)
            new_innerboxes = np.array(boxes[1:, 0:5]).reshape(-1, 5)
    return new_obj_img, new_obj_mask, new_outerbox, new_innerboxes


def recalculate_targetsize(current_targetsize, current_outerbox):
    outerbox = current_outerbox[0, 0:4]
    x1, y1, x2, y2 = outerbox
    bw, bh = (x2 - x1), (y2 - y1)

    if bw % 2 != 0:
        bw += 1
    if bh % 2 != 0:
        bh += 1

    w, h = current_targetsize
    if bw > w and bh > h:
        target_image_size = (bw, bh)
    elif bw > w:
        target_image_size = (bw, h)
    elif bh > h:
        target_image_size = (w, bh)
    else:
        target_image_size = current_targetsize

    return target_image_size


def get_tightfit_targetsize(selected_obj_coords):
    boxes = []
    for key, coord in selected_obj_coords.items():
        boxes.append(coord[0, 0:4])
    boxes = np.array(boxes).reshape(-1, 4)
    x1, y1, x2, y2 = get_outerbox(boxes)[0, 0:4]
    w = x2 - x1
    h = y2 - y1
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    tightfit_targetsize = (w, h)
    return tightfit_targetsize


def get_pastecoords(target_size, all_outer_boxes):
    w, h = target_size
    all_outer_boxes = np.array(all_outer_boxes).reshape(-1, 4)
    wh = all_outer_boxes[..., 2:4] - all_outer_boxes[..., 0:2]
    area = np.reshape(wh[..., 0] * wh[..., 1], (-1,))
    decending_order = np.argsort(-area, axis=-1)
    vertex_pool = [[0, 0]]
    paste_coords = {}

    for i, obj_id in enumerate(decending_order):
        obj_w, obj_h = tuple(wh[obj_id][0:2])
        for j in range(len(vertex_pool)):
            vx, vy = vertex_pool[j]
            obj_newcoord = np.array([vx, vy, vx + obj_w, vy + obj_h]).reshape(-1, 4)
            overlap = False
            if (vx + obj_w) < w and (vy + obj_h) < h:
                for key, prev_paste_coord in paste_coords.items():
                    _, iol, _ = overlap_measure(obj_newcoord, prev_paste_coord)
                    if iol != 0.0:
                        overlap = True
                        break
                if not overlap:
                    paste_coords[obj_id] = obj_newcoord
                    vertex_pool.append([vx + obj_w, vy])
                    vertex_pool.append([vx, vy + obj_h])
                    vertex_pool.append([vx + obj_w, vy + obj_h])
                    del vertex_pool[j]
                    break
    return paste_coords


def get_total_iol(target_image_size, selected_objs_coords):
    total_iol = 0.0
    imgcord = np.array([0, 0, target_image_size[0], target_image_size[1]]).reshape(-1, 4)
    for obj_id, box in selected_objs_coords.items():
        _, iol, _ = overlap_measure(imgcord, box, expand_dim=False)
        total_iol += iol
    return total_iol


def get_realcoords(cutout_objs_coords, cutout_objs_inner_coords, selected_objs_coords):
    realobj_params = []
    for obj_id, new_outer_coord in selected_objs_coords.items():
        old_outer_coord = cutout_objs_coords[obj_id]
        X1, Y1, X2, Y2 = new_outer_coord[0, 0:4]
        x1, y1, x2, y2 = old_outer_coord[0, 0:4]

        vx1 = (X1 - x1)
        vy1 = (Y1 - y1)
        vx2 = (X2 - x2)
        vy2 = (Y2 - y2)
        shift = np.array([vx1, vy1, vx2, vy2]).reshape(-1, 4)
        all_inner_obs = cutout_objs_inner_coords[obj_id]
        all_inner_obs[..., 0:4] += shift
        realobj_params.extend(all_inner_obs)
    realobj_params = np.array(realobj_params).reshape(-1, 5)
    return realobj_params


def paste_masks(targetsize, all_obj_obj_masks, selected_obj_coords):
    new_img_mask = np.zeros((targetsize[1], targetsize[0]), dtype=np.uint8)
    new_inst_id = 1
    for key, coord in selected_obj_coords.items():
        current_mask = np.array(all_obj_obj_masks[key])
        unique_pixel_instances = np.unique(current_mask)
        for inst_id in unique_pixel_instances:
            if inst_id == 0 or inst_id == 255:
                continue
            current_mask[current_mask == inst_id] = new_inst_id
            new_inst_id += 1
        x1, y1, x2, y2 = coord[0, 0:4]
        new_img_mask[y1:y2, x1:x2] = current_mask
    return new_img_mask


def paste_objs(targetsize, all_obj_imgs, selected_obj_coords):
    new_img = np.zeros((targetsize[1], targetsize[0], 3), dtype=np.uint8)
    for key, coord in selected_obj_coords.items():
        x1, y1, x2, y2 = coord[0, 0:4]
        new_img[y1:y2, x1:x2, :] = all_obj_imgs[key]
    return new_img


def simple_paste(src, dst, mask):
    _mask = copy.deepcopy(mask).astype('uint8')
    _mask[_mask != 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    _mask = Image.fromarray(_mask)
    # mask_blur = _mask.filter(ImageFilter.GaussianBlur(3))

    _src = copy.deepcopy(src).astype('uint8')
    _src = Image.fromarray(_src)

    dst = Image.fromarray(dst.astype('uint8'))
    dst = Image.composite(_src, dst, _mask)

    return dst


def seamlessclone(src, dst, mask, center, mode=cv2.NORMAL_CLONE):
    _mask = copy.deepcopy(mask).astype('uint8')
    _mask[_mask != 0] = 255
    dst = cv2.seamlessClone(src, dst, _mask, center, mode)
    dst = Image.fromarray(dst)
    return dst


def paste_at_random_pivot(src_img, src_msk, src_boxes, scale_min=1.0, scale_max=1.5, use_random_backgrnd=True):
    oh, ow = src_img.shape[:2]
    nw = int(rand(scale_min, scale_max) * ow)
    nh = int(rand(scale_min, scale_max) * oh)

    nw = nw + 1 if nw % 2 != 0 else nw
    nh = nh + 1 if nh % 2 != 0 else nh

    px = int(rand(0, (nw - ow)))
    py = int(rand(0, (nh - oh)))

    offsetted_mask = np.zeros((nh, nw), dtype=np.uint8)
    offsetted_mask[py:py + oh, px:px + ow] = src_msk

    offsetted_src_img = np.zeros((nh, nw, 3))
    offsetted_src_img[py:py + oh, px:px + ow] = src_img

    if use_random_backgrnd:
        bcgrnd_color = np.random.randint(0, 255, size=3)
        new_backgrnd = np.array(Image.new(mode="RGB", size=(nw, nh), color=tuple(bcgrnd_color))).astype('uint8')
        offsetted_src_img = simple_paste(offsetted_src_img, new_backgrnd, offsetted_mask)
        offsetted_src_img = np.array(offsetted_src_img)
    src_boxes[..., 0:4] = src_boxes[..., 0:4] + np.array([px, py, px, py]).reshape(-1, 4)
    return offsetted_src_img, offsetted_mask, src_boxes


def mask_to_RGB(mask, colors):
    h, w = mask.height, mask.width
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.array(mask)
    unique_instances = np.unique(mask)
    for p, pxl_instance in enumerate(unique_instances):
        if pxl_instance == 0:
            continue
        elif pxl_instance == 255:
            colored_mask[mask == pxl_instance] = [255, 255, 255]
        else:
            colored_mask[mask == pxl_instance] = colors[p]
    return Image.fromarray(colored_mask)


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def simplest_cb(img, percent):
    """

    From: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    """
    assert img.shape[2] == 3
    assert 0 < percent < 100

    half_percent = percent / 200.0
    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1
        flat = np.sort(flat)
        n_cols = flat.shape[0]
        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


def adjust_color(image, alpha=1.0, beta=0):
    """
    Adjust the brightness and contrast of an image.
    Args:
        image (numpy array): The input image.
        alpha (float): Contrast control (1.0-3.0). 1.0 means no change.
        beta (int): Brightness control (0-100). 0 means no change.

    Returns:
        numpy array: The color-adjusted image.
    """
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image


def adaptive_adjust_color(image, target_brightness=128, target_contrast=64):
    """
    Adjust the image brightness and contrast adaptively based on the histogram.
    Args:
        image (numpy array): The input image.
        target_brightness (int): Desired average brightness of the output image.
        target_contrast (int): Desired standard deviation (contrast) of the output image.

    Returns:
        numpy array: The color-adjusted image.
    """
    # Calculate current brightness and contrast (standard deviation)
    current_brightness = np.mean(image)
    current_contrast = np.std(image)

    # Calculate alpha and beta for adjustment
    if current_contrast > 0:
        alpha = target_contrast / current_contrast
    else:
        alpha = 1.0
    beta = target_brightness - np.mean(image * alpha)

    # Apply the adjustments
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return new_image


def post_processing(image):
    IE = ['HE1', 'HE2', 'ColorBalance']  # 'HE1', ,
    IE_choice = np.random.choice(IE, size=1)
    image = np.array(image)

    if IE_choice == 'HE1':
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    elif IE_choice == 'HE2':
        b_image, g_image, r_image = cv2.split(image)
        b_image_eq = cv2.equalizeHist(b_image)
        g_image_eq = cv2.equalizeHist(g_image)
        r_image_eq = cv2.equalizeHist(r_image)
        image = cv2.merge((b_image_eq, g_image_eq, r_image_eq))
    elif IE_choice == 'ColorBalance':
        image = simplest_cb(image, 5)
    output_img = Image.fromarray(image)
    return output_img


def initialize_directories(args):
    os.makedirs(args.destn_dir, exist_ok=True)
    new_images_dir = os.path.join(args.destn_dir, "augmented_images")
    new_masks_dir = os.path.join(args.destn_dir, "augmented_masks")
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_masks_dir, exist_ok=True)
    return new_images_dir, new_masks_dir


def open_annotation_file(destn_dir):
    new_images_ann_path = os.path.join(destn_dir, 'sigtored_annotations.txt')
    return open(new_images_ann_path, 'w')


def random_new_image_size(width_min_max_pair, height_min_max_weight):
    width_min, width_max = width_min_max_pair
    height_min, height_max = height_min_max_weight
    target_img_width = random.randint(width_min, width_max)
    target_img_height = random.randint(height_min, height_max)
    target_img_width = target_img_width if target_img_width % 2 == 0 else target_img_width + 1
    target_img_height = target_img_height if target_img_height % 2 == 0 else target_img_height + 1
    target_image_size = (target_img_width, target_img_height)

    return target_image_size


def prepare_image_data():
    source_img_path = []
    cutout_objs_images = []
    cutout_objs_masks = []
    cutout_objs_coords = []
    cutout_objs_inner_coords = []
    total_iol = 0.0  # Intersection over largest object area
    selected_objs_coords = []
    return (source_img_path, cutout_objs_images, cutout_objs_masks, cutout_objs_coords, cutout_objs_inner_coords,
            total_iol, selected_objs_coords)


def select_objects_for_image(max_search_iterations, new_img_size, source_ann, source_indx, total_source_anns,
                             maskdir):
    max_iterations = max_search_iterations
    (source_img_path, cutout_objs_images, cutout_objs_masks, cutout_objs_coords, cutout_objs_inner_coords, total_iol,
     selected_objs_coords) = prepare_image_data()

    target_image_size = new_img_size

    while total_iol <= 0.8 or max_iterations > 0:
        source_indx = get_next_source(source_indx, total_source_anns, randomly=False)
        if source_indx == 0:
            np.random.shuffle(source_ann)

        annotation_line = source_ann[source_indx]
        imgpath, obj_img, obj_mask, outerbox, inner_boxes = get_data(annotation_line, maskdir)

        obj_img, obj_mask, outerbox, inner_boxes = random_augmentations(obj_img, obj_mask, outerbox, inner_boxes,
                                                                        max_augs=2)

        source_img_path.append(imgpath)
        cutout_objs_images.append(obj_img)
        cutout_objs_masks.append(obj_mask)
        cutout_objs_coords.append(outerbox)
        cutout_objs_inner_coords.append(inner_boxes)

        target_image_size = recalculate_targetsize(target_image_size, outerbox)
        selected_objs_coords = get_pastecoords(target_image_size, np.array(cutout_objs_coords).reshape(-1, 4))
        total_iol = get_total_iol(target_image_size, selected_objs_coords)
        max_iterations -= 1
    return (source_img_path, cutout_objs_images, cutout_objs_masks, cutout_objs_coords, cutout_objs_inner_coords,
            selected_objs_coords, target_image_size)


def prepare_mask(mask, blur_radius=5, feather_extent=15):
    """
    Prepare the mask by blurring and feathering its edges.
    Args:
        mask (numpy array): The initial mask, should be a single-channel, 255-white for mask, 0-black for background.
        blur_radius (int): Radius for Gaussian blur to smooth the mask.
        feather_extent (int): The extent to which the mask should be feathered.

    Returns:
        numpy array: The prepared mask.
    """
    # Apply Gaussian blur to soften the edges
    blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

    # Feathering by dilating the mask slightly and then blurring
    kernel = np.ones((feather_extent, feather_extent), np.uint8)
    dilated_mask = cv2.dilate(blurred_mask, kernel, iterations=1)
    feathered_mask = cv2.GaussianBlur(dilated_mask, (blur_radius, blur_radius), 0)

    return feathered_mask


def convert_instance_to_binary(mask):
    binary_mask = np.where((mask != 0), 255, 0).astype(np.uint8) # & (mask != 255)

    return binary_mask


def create_composite_image(source_img, destn_image, mask):
    center = (source_img.shape[1] // 2, source_img.shape[0] // 2)
    clone_choice = np.random.choice(['SP'])  #, 'Normal', 'Mixed', 'Monochrome_Transfer'
    if clone_choice == 'SP':
        composite_image = simple_paste(source_img, destn_image, mask)
    elif clone_choice == 'NormalClone':
        composite_image = seamlessclone(source_img, destn_image, mask, center, mode=cv2.NORMAL_CLONE)
    elif clone_choice == 'MixedClone':
        composite_image = seamlessclone(source_img, destn_image, mask, center, mode=cv2.MIXED_CLONE)
    else:
        composite_image = seamlessclone(source_img, destn_image, mask, center, mode=cv2.MONOCHROME_TRANSFER)

    return composite_image


def compose_final_image(args, selected_objs_coords, cutout_objs_coords, cutout_objs_inner_coords, cutout_objs_images,
                        cutout_objs_masks):
    target_image_size = get_tightfit_targetsize(selected_objs_coords)
    # Stitch the selected objects masks
    new_mask = paste_masks(target_image_size, cutout_objs_masks, selected_objs_coords)
    # Stitch the selected objects images
    new_img = paste_objs(target_image_size, cutout_objs_images, selected_objs_coords)

    # Blur and erode the mask for composing the artificial image with minimal artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_mask = cv2.erode(new_mask, kernel, iterations=2)

    # Get the real tight-fit coordinates of the objects
    new_boxes1 = get_realcoords(cutout_objs_coords, cutout_objs_inner_coords, selected_objs_coords)
    # Paste the selected objects on the new artificial image on a random coordinate instead of pasting always at (0,0)
    new_img2, new_mask2, new_boxes2 = paste_at_random_pivot(new_img, new_mask, new_boxes1,
                                                            scale_min=1.0, scale_max=1.15)

    # For seamless cloning first convert the mask into binary mask, black and white
    mask = convert_instance_to_binary(new_mask2)

    # Pick a background image to compose the new image
    bcgrnd_img = get_backgrnd_image(args.bckgrnd_imgs_dir, (new_img2.shape[1], new_img2.shape[0], 3))

    # Preprocess the image created by stitching objects to enhance brightness
    new_img2 = adaptive_adjust_color(new_img2)
    # Apply the same preprocessing to the background image as well
    bcgrnd_img = adaptive_adjust_color(bcgrnd_img)

    # create the composite image using random cloning choice
    new_artificial_img = create_composite_image(new_img2, bcgrnd_img, mask)

    # Optionally do some final post-processing, such as brightness or contrast enhancement
    final_image = post_processing(new_artificial_img)

    # Convert the unprocessed mask of the artificial image into color (RGB) for segmentation mask creation
    new_mask2 = Image.fromarray(new_mask2)
    final_colored_mask = mask_to_RGB(new_mask2, get_colors(256))
    final_boxes = new_boxes2

    return final_image, final_colored_mask, final_boxes


def save_new_image_and_mask(final_image, final_mask, final_boxes, new_dataset, new_images_dir, new_masks_dir,
                            count_new_images):
    newimgpath = os.path.join(new_images_dir, "{:08d}.jpg".format(count_new_images))
    final_image.save(newimgpath)

    newmaskpath = os.path.join(new_masks_dir, "{:08d}.png".format(count_new_images))
    final_mask.save(newmaskpath)

    annotation_line = convert_to_ann_line(newimgpath, final_boxes)
    new_dataset.write(annotation_line)
    new_dataset.flush()


def main(args):
    source_ann = read_ann(args.source_ann_file)  # All ground-truth annotations
    maskdir = args.mask_image_dirs

    os.makedirs(args.destn_dir, exist_ok=True)
    new_images_dir = os.path.join(args.destn_dir, "augmented_images")
    new_masks_dir = os.path.join(args.destn_dir, "augmented_masks")
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_masks_dir, exist_ok=True)
    new_images_ann_path = os.path.join(args.destn_dir, 'sigtored_annotations.txt')
    new_dataset = open(new_images_ann_path, 'w')

    total_source_anns = len(source_ann)
    source_indx = -1

    count_new_images = 0
    max_search_iterations = 3
    for _ in tqdm(range(args.total_new_imgs), desc='Generating artificial images', leave=True):
        target_image_size = random_new_image_size((400, 600), (400, 600))

        (source_img_path, cutout_objs_images, cutout_objs_masks, cutout_objs_coords, cutout_objs_inner_coords,
         selected_objs_coords, target_image_size) = select_objects_for_image(
            max_search_iterations, target_image_size, source_ann, source_indx, total_source_anns, args.mask_image_dirs)

        final_image, final_mask, final_boxes = compose_final_image(args, selected_objs_coords, cutout_objs_coords,
                                                                   cutout_objs_inner_coords, cutout_objs_images,
                                                                   cutout_objs_masks)

        save_new_image_and_mask(final_image, final_mask, final_boxes, new_dataset, new_images_dir, new_masks_dir,
                                count_new_images)

        count_new_images += 1


if __name__ == '__main__':
    config_path = "./sig_argument.txt"
    if os.path.exists(config_path):
        arguments = parse_commandline_arguments(config_path)
    else:
        raise ValueError("path {} containing general SIGtor arguments is not found.".format(config_path))

    parser = argparse.ArgumentParser(
        description='Supplementary Synthetic Image Generation for Object Detection and Segmentation')
    parser.add_argument('--source_ann_file', type=str, required=False, default=arguments['source_ann_file'][0],
                        help='YOLO format annotation txt file as a source dataset')
    parser.add_argument('--destn_dir', type=str, required=False, default=arguments['destn_dir'][0],
                        help='directory to save the generated images, their ground_truth annotations and masks')
    parser.add_argument('--mask_image_dirs', type=str, required=False, default=arguments['mask_image_dirs'][0],
                        help='directory of where to find the ground-truth masks, if there are any')
    parser.add_argument('--bckgrnd_imgs_dir', type=str, required=False, default=arguments['bckgrnd_imgs_dir'][0],
                        help='directory of where the background images are found if there are any')
    parser.add_argument('--total_new_imgs', type=int, required=False, default=int(arguments['total_new_imgs'][0]),
                        help='total number of new images to generate from the total annotations.')
    args = parser.parse_args()
    main(args)
