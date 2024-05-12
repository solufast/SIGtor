
# SIGtor: Supplementary Synthetic Image Generation for Object Detection and Segmentation Datasets

## Introduction

In deep learning, tasks such as classification, detection, and segmentation often face the challenge of requiring a large, well-balanced training dataset. Generating substantial datasets for detection and segmentation is particularly challenging, as it is both tedious and prone to errors. Consequently, data augmentation has become crucial in training deep learning models, allowing small datasets to be augmented through morphological or geometrical modifications, either on-the-fly during training or offline.

SIGtor introduces a method for artificially generating an almost infinite number of new, supplementary datasets for object detection or segmentation from any existing dataset. This robust, copy-paste based augmentation approach handles object overlapping, dynamic placement on various background images, and supports both object-level and image-wide augmentations. The generated images include instance segmentation masks and tightly fitted bounding boxes.

## Prerequisites

1. **Existing Dataset**: You must have a dataset that you wish to extend using SIGtor. Note that SIGtor is an offline dataset generator, not a real-time augmentation tool used during model training. The generated images can be utilized with or without the original images for training.
   
2. **Annotation Format**: Your dataset should be annotated in YOLO format, with bounding boxes specified as follows:  
   `./Datasets/Source/Images/image1.jpg x1,y1,x2,y2,A x1,y1,x2,y2,B x1,y1,x2,y2,C x1,y1,x2,y2,D`  
   For demonstrations, this project will utilize datasets in Pascal VOC or COCO formats, which can be converted to YOLO format using tools provided in the project's tools folder.

3. **Background Images**: Download suitable background images for use in synthetic image generation. Store these images in a `BackgroundImages` directory, ensuring they do not contain objects from your dataset classes. Automated download scripts and manual curation steps are recommended to avoid introducing unannotated objects into your training dataset.

## Workflow

### Step 1: Expand Source Annotations

Expand the source annotations to calculate the Intersection over Union (IoU) among objects, re-annotating to manage overlaps and containments effectively. The `expand_annotation.py` script facilitates this expansion. For instance, separate annotations are generated for non-overlapping objects or objects completely within another object. 

### Step 2: Generate Artificial Images

The synthetic images are generated using the `synthetic_image_generator.py` script. Detailed steps and adjustments can be found in the `sig_argument.txt` file, guiding the generation process to suit specific dataset requirements or different object detection and segmentation challenges.

## Detailed Workflow of SIGtor
SIGtor operates by enriching object detection or segmentation datasets through a sophisticated image generation process, accommodating datasets with images that include object masks (preferably instance segmentation masks) and optional background images. Here’s how the algorithm processes each dataset:

1. **Input Selection**: SIGtor begins by randomly selecting a source image along with its corresponding mask from the dataset. If no mask is available, one is created using the bounding box coordinates of the objects in the image.
2. **Background Preparation**: A target background image is chosen randomly and resized to fit a predetermined dimension.
3. **Object Selection**: The algorithm then selects an object cutout from the source image, including its coordinates and mask. If the object contains nested objects, these are also included. All objects pass through various geometric and morphological augmentation, object-wise augmentation, before pasted to a new background.
4. **Overlap Management**: This step is repeated until the Sum of Intersection over Larger Image (IoL, which is going to be the IoL of cutout objects and the background image in this case) of the selected objects surpasses a set threshold (e.g., 80%), ensuring higher coverage of the background image with cutout objects.
5. **Object Placement**: All selected objects are strategically placed on the background, ensuring optimal space utilization and no overlap. The same placement strategy applies to the objects' masks. 
6. **Image Composition**: Using techniques such as seamless cloning, alpha blending, or simple copy-pasting, SIGtor creates a new composite image. It also generates an instance segmentation mask for this composite and prepares a new annotation line.
7. **Saving Outputs**: The composite image, its mask, and the new annotations are saved.
8. **Post Processing**: The composite image once again passes through some random image wide augmentation.
9. **Iteration**: Steps 1 through 7 are repeated until the desired number of synthetic images is generated.
This workflow efficiently utilizes background and object data to create diverse training samples, enhancing the depth and variety of the dataset without needing to acquire new images. This process not only saves resources but also introduces variability in a controlled manner, essential for training robust detection and segmentation models.

## Conclusion

SIGtor has significantly enhanced the training capabilities of models like YOLOv3 and its derivatives through sophisticated copy-paste augmentation. It is important to manage the diversity and repetition in the training set to prevent overfitting and to ensure balanced representation of object classes. Although some artifacts from synthetic images may remain, they generally do not adversely affect model performance, unless the training duration extends sufficiently for the model to begin overfitting to these details.
