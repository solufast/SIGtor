import argparse
import os

import cv2
import numpy as np

from utils import read_ann, get_colors, draw_boxes, parse_commandline_arguments, get_classes


def visualize_sigtor(ann_path, class_names=None):
    lines = read_ann(ann_path, shuffle=True)
    for line in lines:
        line_content = line.split()
        img_path = line_content[0]
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line_content[1:]]).astype('int32').reshape(
            -1, 5)
        classes = boxes[..., 4] + 1
        if class_names is None:
            class_names = classes
        colors = get_colors()
        image = draw_boxes(image, boxes[..., 0:4], classes, class_names, colors)
        cv2.imwrite("misc/images/{}".format(filename), image)


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
    parser.add_argument('--classnames_file', type=str, required=False, default=arguments['classnames_file'][0],
                        help='Dataset object classes')

    args = parser.parse_args()
    class_names = get_classes(args.classnames_file)
    visualize_sigtor(args.source_ann_file, class_names)
