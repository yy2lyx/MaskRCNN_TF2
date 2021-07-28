import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import random
import numpy as np
from draw_dotted_line import drawrect


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

CLASS_NAMES = ['BG', 'pig','cow']
SHOW_BBOX = True
SHOW_MASK = True

class AnimalConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "animal"

    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)


# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference",
                             config=AnimalConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="logs/animal20210727T1506/mask_rcnn_animal_0022.h5",
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("0.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]



colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASS_NAMES))]

masked_image = image.copy()


target = r['rois'].shape[0]
ids = r['class_ids']
scores = r['scores']
masks = r['masks']

for i in range(target):
    color_i = colors[ids[i]]
    if not np.any(r['rois'][i]):
        # 如果没有bbox，直接跳过
        continue
    if scores[i] <= 0.8:
        continue
    # 1.画虚线框，bbox
    y1, x1, y2, x2 = r['rois'][i]

    if SHOW_BBOX:
        drawrect(masked_image,(x1,y1),(x2,y2),color_i,thickness=7)

    # 2. 画mask部分
    mask = masks[:, :, i]
    a = np.where()
    if SHOW_MASK:
        # 给每个点上色
        masked_image = apply_mask(masked_image, mask, color_i)
        # 给每个实例画出轮廓边缘
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2),dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask

        # 找到轮廓
        contours, hierarchy = cv2.findContours(padded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 画出轮廓
        for i in range(len(contours)):
            cv2.drawContours(masked_image, contours, i, color_i, 10, cv2.LINE_8, hierarchy, 0)



cv2.namedWindow("1",cv2.WINDOW_NORMAL)
cv2.imshow("1",masked_image)
cv2.waitKey(0)



# mrcnn.visualize.display_instances(image=image,
#                                   boxes=r['rois'],
#                                   masks=r['masks'],
#                                   class_ids=r['class_ids'],
#                                   class_names=CLASS_NAMES,
#                                   scores=r['scores'])