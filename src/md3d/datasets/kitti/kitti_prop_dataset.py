import fnmatch
import os
import pickle

import md3d
import numpy as np

from md3d.datasets.kitti import obj_utils
from md3d.datasets.kitti.obj_utils import Difficulty


class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs

    def __repr__(self):
        return '({}, augs: {})'.format(self.name, self.augs)


class KittiPropDataset:

    def __init__(self, dataset_config, train_val_test):

        self.dataset_config = dataset_config
        self.train_val_test = train_val_test

        # Parse config
        self.name = self.dataset_config.name

        self.data_split = self.dataset_config.data_split
        self.dataset_dir = os.path.expanduser(self.dataset_config.dataset_dir)
        data_split_dir = self.dataset_config.data_split_dir

        # self.num_boxes = self.dataset_config.num_boxes

        self.num_ang_bins = self.dataset_config.num_angle_bins
        self.ang_bin_overlap = self.dataset_config.angle_bin_overlap

        self.roi_shape = [48, 48]

        # TODO: set up both cameras if possible
        self.cam_idx = 2

        self.obj_types = self.dataset_config.obj_types
        self.num_obj_types = len(self.obj_types)

        # Object filtering config
        if self.train_val_test in ['train', 'val']:
            obj_filter_config = self.dataset_config.obj_filter_config
            obj_filter_config.obj_types = self.obj_types
            self.obj_filter = obj_utils.ObjectFilter(obj_filter_config)

        else:  # self.train_val_test == 'test'
            # Use all detections during inference
            self.obj_filter = obj_utils.ObjectFilter.create_obj_filter(
                obj_types=self.obj_types,
                difficulty=Difficulty.ALL,
                occlusion=None,
                truncation=None,
                box_2d_height=None,
                depth_range=None)

        self.has_kitti_labels = self.dataset_config.has_kitti_labels

        # Always use statistics computed using KITTI 2D boxes
        self.trend_data = 'kitti'

        self.classes_name = self._set_up_classes_name()

        if self.classes_name == 'Car':
            self.mscnn_merge_min_iou = 0.7
        elif self.classes_name in ['Pedestrian', 'Cyclist']:
            self.mscnn_merge_min_iou = 0.5

        # Check that paths and split are valid
        self._check_dataset_dir()
        all_dataset_files = os.listdir(self.dataset_dir)
        self._check_data_split_valid(all_dataset_files)
        self.data_split_dir = self._check_data_split_dir_valid(
            all_dataset_files, data_split_dir)
        self.data_split_dir_name = data_split_dir

        self.depth_version = self.dataset_config.depth_version
        self.instance_version = self.dataset_config.instance_version

        # Setup directories
        self._set_up_directories()

        # Augmentation
        self.aug_config = self.dataset_config.aug_config

        self.sample_names = self.load_sample_names(self.data_split)
        self.num_samples = len(self.sample_names)

        # Batch pointers
        self._sample_order = np.arange(self.num_samples)
        self._index_in_epoch = 0
        self.epochs_completed = 0

    def _check_dataset_dir(self):
        """Checks that dataset directory exists in the file system

        Raises:
            FileNotFoundError: if the dataset folder is missing
        """
        # Check that dataset path is valid
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError('Dataset path does not exist: {}'
                                    .format(self.dataset_dir))

    def _check_data_split_valid(self, all_dataset_files):
        possible_splits = []
        for file_name in all_dataset_files:
            if fnmatch.fnmatch(file_name, '*.txt'):
                possible_splits.append(os.path.splitext(file_name)[0])
        # This directory contains a readme.txt file, remove it from the list
        if 'readme' in possible_splits:
            possible_splits.remove('readme')

        if self.data_split not in possible_splits:
            raise ValueError("Invalid data split: {}, possible_splits: {}"
                             .format(self.data_split, possible_splits))

    def _check_data_split_dir_valid(self, all_dataset_files, data_split_dir):
        # Check data_split_dir
        # Get possible data split dirs from folder names in dataset folder
        possible_split_dirs = []
        for folder_name in all_dataset_files:
            if os.path.isdir(self.dataset_dir + '/' + folder_name):
                possible_split_dirs.append(folder_name)

        if data_split_dir in possible_split_dirs:
            # Overwrite with full path
            data_split_dir = self.dataset_dir + '/' + data_split_dir
        else:
            raise ValueError(
                "Invalid data split dir: {}, possible dirs".format(
                    data_split_dir, possible_split_dirs))

        return data_split_dir

    def _set_up_directories(self):
        """Sets up data directories."""
        # Setup Directories
        self.rgb_image_dir = self.data_split_dir + '/image_' + str(self.cam_idx)
        self.image_2_dir = self.data_split_dir + '/image_2'
        self.image_3_dir = self.data_split_dir + '/image_3'

        self.calib_dir = self.data_split_dir + '/calib'
        self.disp_dir = self.data_split_dir + '/disparity'
        self.planes_dir = self.data_split_dir + '/planes'
        self.velo_dir = self.data_split_dir + '/velodyne'
        self.depth_dir = self.data_split_dir + '/depth_{}_{}'.format(
            self.cam_idx, self.depth_version)

        self.depth_2_dir = self.data_split_dir + '/depth_2_{}'.format(self.depth_version)
        self.depth_3_dir = self.data_split_dir + '/depth_3_{}'.format(self.depth_version)

        if self.has_kitti_labels:
            self.kitti_label_dir = self.data_split_dir + '/label_2'

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_obj_types > 1:
            raise NotImplementedError('Number of classes must be 1')
        else:
            classes_name = self.obj_types[0]

        return classes_name

    def use_same_calib(self):
        if not self.is_using_same_calib:
            self.rgb_image_dir += '_same_calib'
            self.image_2_dir += '_same_calib'
            self.calib_dir += '_same_calib'
            self.kitti_label_dir += '_same_calib'
            self.depth_dir += '_same_calib'
            self.instance_dir += '_same_calib'

            self.is_using_same_calib = True

    # def get_sample_names(self):
    #     return [sample.name for sample in self.sample_list]

    # Get sample paths
    def get_rgb_image_path(self, sample_name):
        return self.rgb_image_dir + '/' + sample_name + '.png'

    def get_image_2_path(self, sample_name):
        return self.image_2_dir + '/' + sample_name + '.png'

    def get_image_3_path(self, sample_name):
        return self.image_3_dir + '/' + sample_name + '.png'

    # def get_depth_map_path(self, sample_name):
    #     return self.depth_dir + '/' + sample_name + '_left_depth.png'

    def get_velodyne_path(self, sample_name):
        return self.velo_dir + '/' + sample_name + '.bin'

    # Cluster info
    def get_cluster_info(self):
        return self.clusters, self.std_devs

    # Data loading methods
    def load_sample_names(self, data_split):
        """Load the sample names listed in this dataset's set file
        (e.g. train.txt, validation.txt)

        Args:
            data_split: the sample list to load

        Returns:
            A list of sample names (file names) read from
            the .txt file corresponding to the data split
        """
        set_file = self.dataset_dir + '/' + data_split + '.txt'
        with open(set_file, 'r') as f:
            sample_names = f.read().splitlines()

        return np.asarray(sample_names)

    def load_samples(self, indices):
        """ Loads input-output data for a set of samples. Should only be
            called when a particular sample dict is required. Otherwise,
            samples should be provided by the next_batch function

        Args:
            indices: A list of sample indices from the dataset.sample_list
                to be loaded

        Return:
            samples: a list of data sample dicts
        """

        sample_idx = indices[0]

        sample_name = self.sample_names[sample_idx]

        # Load lidar point cloud
        # frame_calib = calib_utils.get_frame_calib(self.calib_dir, sample_name)
        # lidar_pc, lidar_i = obj_utils.get_lidar_point_cloud(
        #     sample_name, frame_calib, self.velo_dir + '_reduced', intensity=True)

        # Load depth point cloud
        # cam_p = frame_calib.p2
        # depth_pc = obj_utils.get_depth_map_point_cloud(
        #     sample_name, cam_p, self.depth_dir)

        # Load image
        bgr_image = obj_utils.get_image(sample_name, self.image_2_dir)
        rgb_image = bgr_image[..., ::-1]

        if self.train_val_test in ['train', 'val']:
            # Load props
            prop_objs_info_dir = md3d.data_dir() + '/prop_objs_info/{}/train_mode/{}x{}'.format(
                self.data_split_dir_name, self.roi_shape[0], self.roi_shape[1])
        else:
            raise ValueError('Not supported yet')

        prop_objs_info_path = prop_objs_info_dir + '/{}.pkl'.format(sample_name)

        with open(prop_objs_info_path, 'rb') as f:
            prop_objs_info = pickle.load(f)

        # Return None if no ground truth
        # TODO: randomly decide whether to keep
        if self.train_val_test in ['train', 'val']:
            if not prop_objs_info['has_positives']:
                return None
        else:
            raise ValueError('Not supported yet')

        sample_dict = {
            'sample_name': sample_name,

            'image': rgb_image,
            'prop_objs_info': prop_objs_info,
        }

        return sample_dict

    def _shuffle_order(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self._sample_order = perm

    def next_batch(self, batch_size, shuffle):
        """
        Retrieve the next `batch_size` samples from this data set.

        Args:
            batch_size: number of samples in the batch
            shuffle: whether to shuffle the indices after an epoch is completed

        Returns:
            list of dictionaries containing sample information
        """

        start = self._index_in_epoch
        # Shuffle only for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle_order()

        if start + batch_size >= self.num_samples:

            # Finished epoch
            self.epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.num_samples - start

            # Append those samples to the current batch
            sample_range = np.arange(start, self.num_samples)
            rand_indices = self._sample_order.take(sample_range)

            # Shuffle the data
            if shuffle:
                self._shuffle_order()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            # Append the rest of the batch
            sample_range = np.arange(start, end)
            rand_indices = np.append(rand_indices, self._sample_order.take(sample_range))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # Append the samples in the range to the batch
            sample_indices = np.arange(start, end)
            rand_indices = self._sample_order.take(sample_indices)

        samples_in_batch = self.load_samples(rand_indices)

        return samples_in_batch
