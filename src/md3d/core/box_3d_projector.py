"""
Projects boxes into bird's eye view and image space.
Returns the 4 points (x, y) of the corresponding box
"""
import numpy as np

from md3d.datasets.kitti import calib_utils
from md3d.datasets.kitti import obj_utils

from md3d.core import box_3d_encoder


def project_to_bev(boxes_3d, bev_extents):
    """
    Projects an array of 3D boxes into bird's eye view

    Args:
        boxes_3d: list of 3d boxes in the format:
            N x [x, y, z, l, w, h, ry]
        bev_extents: xz extents of the 3d area
            [[min_x, max_x], [min_z, max_z]]

    Returns:
        box_points: counter-clockwise order box points in bev map space
            N x [[x0, y0], ... [x3, y3]] - (N x 4 x 2)
        box_points_norm: points normalized as a percentage of the map size
            N x [[x0, y0], ... [x3, y3]] - (N x 4 x 2)
    """

    boxes_3d = np.array(boxes_3d, dtype=np.float32)
    x = boxes_3d[:, 0]
    z = boxes_3d[:, 2]
    l = boxes_3d[:, 3]
    w = boxes_3d[:, 4]
    ry = boxes_3d[:, 6]

    # 1|0 2D corners
    # 2|3
    l_2 = l / 2.0
    w_2 = w / 2.0

    p0 = np.array([l_2, w_2])
    p1 = np.array([-l_2, w_2])
    p2 = np.array([-l_2, -w_2])
    p3 = np.array([l_2, -w_2])

    box_points = np.empty((len(boxes_3d), 4, 2))

    for box_idx in range(len(boxes_3d)):
        rot = ry[box_idx]

        rot_mat = np.reshape([[np.cos(rot), np.sin(rot)],
                              [-np.sin(rot), np.cos(rot)]],
                             (2, 2))

        box_x = x[box_idx]
        box_z = z[box_idx]

        box_xz = [box_x, box_z]

        box_p0 = np.dot(rot_mat, p0[:, box_idx]) + box_xz
        box_p1 = np.dot(rot_mat, p1[:, box_idx]) + box_xz
        box_p2 = np.dot(rot_mat, p2[:, box_idx]) + box_xz
        box_p3 = np.dot(rot_mat, p3[:, box_idx]) + box_xz

        box_points[box_idx] = np.array([box_p0, box_p1, box_p2, box_p3])

    # Calculate normalized box corners for ROI pooling
    x_extents_min = bev_extents[0][0]
    z_extents_min = bev_extents[1][1]  # z axis is reversed
    points_shifted = box_points - [x_extents_min, z_extents_min]

    x_extents_range = bev_extents[0][1] - bev_extents[0][0]
    z_extents_range = bev_extents[1][0] - bev_extents[1][1]
    box_points_norm = points_shifted / [x_extents_range, z_extents_range]

    box_points = np.asarray(box_points, dtype=np.float32)
    box_points_norm = np.asarray(box_points_norm, dtype=np.float32)

    return box_points, box_points_norm


def project_boxes_3d_to_image_space(boxes_3d, cam_p, truncate=False, image_shape=None,
                                    discard=True, discard_before_truncation=True):

    return np.asarray([project_to_image_space(
        det_box_3d, cam_p, truncate=truncate, image_shape=image_shape,
        discard=discard, discard_before_truncation=discard_before_truncation)
        for det_box_3d in boxes_3d])


def project_to_image_space(box_3d, cam_p,
                           truncate=False, image_shape=None,
                           discard=True, discard_before_truncation=True):
    """ Projects a box_3d into image space

    Args:
        box_3d: single box_3d to project
        cam_p: (3, 4) camera projection matrix
        truncate: if True, 2D projections are truncated to be inside the image
        image_shape: [h, w] must be provided if truncate is True,
            used for truncation
        discard: if True, discard boxes that are truncated over a certain amount
        discard_before_truncation: If True, discard boxes that are larger than
            80% of the image in width OR height BEFORE truncation. If False,
            discard boxes that are larger than 80% of the width AND
            height AFTER truncation.

    Returns:
        Projected box in image space [y1, x1, y2, x2]
            Returns None if box is not inside the image
    """

    obj_label = box_3d_encoder.box_3d_to_object_label(box_3d)
    corners_3d = obj_utils.compute_obj_label_corners_3d(obj_label)

    projected = calib_utils.project_pc_to_image(corners_3d, cam_p)

    x1 = np.amin(projected[0])
    y1 = np.amin(projected[1])
    x2 = np.amax(projected[0])
    y2 = np.amax(projected[1])

    img_box_2d = np.array([x1, y1, x2, y2])

    if truncate:
        if not image_shape:
            raise ValueError('Image size must be provided')

        image_h, image_w = image_shape

        # Discard invalid boxes (outside image space)
        if img_box_2d[0] > image_w or \
                img_box_2d[1] > image_h or \
                img_box_2d[2] < 0 or \
                img_box_2d[3] < 0:
            return [0, 0, 1, 1]

        # Discard boxes that are larger than 80% of the image width OR height
        if discard and discard_before_truncation:
            img_box_w = img_box_2d[2] - img_box_2d[0]
            img_box_h = img_box_2d[3] - img_box_2d[1]
            if img_box_w > (image_w * 0.8) or img_box_h > (image_h * 0.8):
                return [0, 0, 1, 1]

        # Truncate remaining boxes into image space
        if img_box_2d[0] < 0:
            img_box_2d[0] = 0
        if img_box_2d[1] < 0:
            img_box_2d[1] = 0
        if img_box_2d[2] > image_w:
            img_box_2d[2] = image_w
        if img_box_2d[3] > image_h:
            img_box_2d[3] = image_h

        # Discard boxes that are covering the the whole image after truncation
        if discard and not discard_before_truncation:
            img_box_w = img_box_2d[2] - img_box_2d[0]
            img_box_h = img_box_2d[3] - img_box_2d[1]
            if img_box_w > (image_w * 0.8) and img_box_h > (image_h * 0.8):
                return [0, 0, 1, 1]

    return img_box_2d[[1, 0, 3, 2]]
