import cv2
import numpy as np


def read_instance_image(instance_image_path):
    instance_image = cv2.imread(instance_image_path, cv2.IMREAD_GRAYSCALE)
    return instance_image


def get_instance_image(sample_name, instance_dir):

    instance_image_path = instance_dir + '/{}.png'.format(sample_name)
    instance_image = read_instance_image(instance_image_path)

    return instance_image


def get_instance_mask_list(instance_img, num_instances=None):
    """Creates n-dimensional image from instance image with one channel per instance

    Args:
        instance_img: (H, W) instance image
        num_instances: (optional) number of instances in the image. If None, will use the
            highest value pixel as the number of instances, but may miss the last
            instances if they have no points.

    Returns:
        instance_masks: (k, H, W) instance masks where k is the unique values of the instance im
    """

    if num_instances is None:
        valid_pixels = instance_img[instance_img != 255]
        if len(valid_pixels) == 0:
            return []
        num_instances = np.max(valid_pixels) + 1

    instance_masks = np.asarray([(instance_img == instance_idx)
                                 for instance_idx in range(num_instances)])
    return instance_masks


def read_instance_maps(instance_maps_path):
    return np.load(instance_maps_path)
