import os
import sys

import cv2
import numpy as np

import md3d
from md3d.builders.dataset_builder import DatasetBuilder
from md3d.datasets.kitti import obj_utils
from md3d.datasets.kitti.kitti_prop_dataset import KittiPropDataset
from md3d.datasets.kitti.obj_utils import Difficulty


def main(split_idx=None, num_splits=None):

    ##############################
    # Options
    ##############################
    dataset = KittiPropDataset(
        DatasetBuilder.get_config_obj(
            DatasetBuilder.KITTI_TRAINVAL),
        train_val_test='train')

    classes = ['Pedestrian']
    difficulty = Difficulty.ALL

    output_dir = md3d.data_dir() + '/outputs/roi'
    os.makedirs(output_dir, exist_ok=True)

    ##############################
    # End of Options
    ##############################

    if split_idx is None:
        samples_to_use = dataset.sample_names
    else:
        samples_split = np.array_split(dataset.sample_names, num_splits)
        samples_to_use = samples_split[split_idx]

    for sample_name in samples_to_use:

        # Get bounding boxes
        all_gt_obj_labels = obj_utils.read_labels(dataset.kitti_label_dir, sample_name)
        gt_obj_labels, _ = obj_utils.filter_labels(
            all_gt_obj_labels,
            classes=classes,
            difficulty=difficulty)

        print(sample_name)

        if len(gt_obj_labels) > 0:

            # Get sample info
            image = obj_utils.get_image(sample_name, dataset.image_2_dir)
            gt_boxes_2d = obj_utils.boxes_2d_from_obj_labels(gt_obj_labels)

            for obj_idx in range(len(gt_obj_labels)):
                gt_box_2d = gt_boxes_2d[obj_idx]

                roi_y1 = int(np.round(gt_box_2d[0]))
                roi_x1 = int(np.floor(gt_box_2d[1]))
                roi_y2 = int(np.round(gt_box_2d[2]))
                roi_x2 = int(np.ceil(gt_box_2d[3]))

                roi_crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_resized = cv2.resize(roi_crop, (224, 224))

                # cv2.imshow('image', image)
                # cv2.imshow('roi_resized', roi_resized)
                # cv2.waitKey(0)

                output_path = output_dir + '/{}_{:02d}.png'.format(sample_name, obj_idx)
                cv2.imwrite(output_path, roi_resized)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        split_idx = int(sys.argv[1])
        num_splits = int(sys.argv[2])
        print('split / num_splits', split_idx, num_splits)
        main(split_idx, num_splits)
    else:
        main()
