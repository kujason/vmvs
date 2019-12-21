import cv2
import os
import sys
import time

import numpy as np
import vtk
from vtk.util import numpy_support

import md3d
from md3d.builders.dataset_builder import DatasetBuilder
from md3d.datasets.kitti import obj_utils, calib_utils
from md3d.datasets.kitti.kitti_prop_dataset import KittiPropDataset
from scene_vis.vtk_wrapper.vtk_point_cloud import VtkPointCloud


def main():

    ##############################
    # Options
    ##############################

    # offscreen_rendering = True
    offscreen_rendering = False  # show render grid

    # Area extents
    x_min = -40
    x_max = 40
    y_min = -5
    y_max = 5
    z_min = 0
    z_max = 70

    dataset = KittiPropDataset(
        DatasetBuilder.get_config_obj(
            DatasetBuilder.KITTI_TRAINVAL),
        train_val_test='train')

    label_dir = dataset.kitti_label_dir

    max_num_objs = 4
    num_cam_pos = 8

    classes = ['Pedestrian']

    roi_size = 224

    ##############################
    # End of Options
    ##############################

    def vtkImageData_to_np(vtkImageData):
        scalars = vtkImageData.GetPointData().GetScalars()
        np_out = numpy_support.vtk_to_numpy(scalars)
        np_out = np_out.reshape(roi_size * max_num_objs, roi_size * num_cam_pos, -1)[::-1, :, ::-1]
        return np_out

    samples_to_use = dataset.sample_names
    area_extents = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]], dtype=np.float32)

    ##############################
    # Vtk Visualization
    ##############################
    vtk_pc_depth = VtkPointCloud()
    vtk_pc_depth.vtk_actor.GetProperty().SetPointSize(4)

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName("Obj Point Cloud")
    vtk_render_window.SetOffScreenRendering(offscreen_rendering)
    vtk_render_window.SetSize(roi_size * num_cam_pos, roi_size * max_num_objs)

    # Create Renderers
    all_vtk_renderers = np.full([max_num_objs, num_cam_pos], None)
    viewport_width = 1.0 / num_cam_pos
    viewport_height = 1.0 / max_num_objs
    for obj_idx in range(max_num_objs):
        for cam_pos_idx in range(num_cam_pos):
            vtk_renderer = vtk.vtkRenderer()
            vtk_renderer.SetBackground(0.2, 0.3, 0.4)
            vtk_renderer.AddActor(vtk_pc_depth.vtk_actor)

            vtk_renderer.SetViewport(cam_pos_idx * viewport_width,
                                     obj_idx * viewport_height,
                                     (cam_pos_idx + 1) * viewport_width,
                                     (obj_idx + 1) * viewport_height)

            vtk_render_window.AddRenderer(vtk_renderer)

            all_vtk_renderers[obj_idx][cam_pos_idx] = vtk_renderer

            # Setup Camera
            current_cam = vtk_renderer.GetActiveCamera()
            current_cam.Roll(180.0)

    # Initialize
    vtk_render_window.Render()

    for sample_idx, sample_name in enumerate(samples_to_use):

        # Get bounding boxes
        all_gt_obj_labels = obj_utils.read_labels(label_dir, sample_name)[0:max_num_objs]
        gt_obj_labels, _ = obj_utils.filter_labels(
            all_gt_obj_labels,
            classes=classes)

        # Get renders
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(vtk_render_window)

        print('{} / {}'.format(sample_idx, len(samples_to_use) - 1))

        if len(gt_obj_labels) > 0:

            # Get sample info
            image = obj_utils.get_image(sample_name, dataset.image_2_dir)

            # image_shape = image.shape[0:2]
            frame_calib = calib_utils.get_frame_calib(dataset.calib_dir, sample_name)
            cam_p = frame_calib.p2

            # Get points from depth map
            depth_point_cloud = obj_utils.get_depth_map_point_cloud(
                sample_name, cam_p, dataset.depth_dir)
            depth_point_cloud, area_filter = obj_utils.filter_pc_to_area(
                depth_point_cloud, area_extents)

            # Filter depth map points to area
            area_filter = np.reshape(area_filter, image.shape[0:2])
            depth_point_colours = image[area_filter]

            total_time = time.time()
            depth_pc_time = time.time()
            vtk_pc_depth.set_points(depth_point_cloud.T, depth_point_colours)
            print('depth_pc_time', time.time() - depth_pc_time)

            num_gt_objs = len(gt_obj_labels)
            for obj_idx in range(max_num_objs):

                if obj_idx >= num_gt_objs:
                    empty_renderers = all_vtk_renderers[obj_idx]
                    [r.DrawOff() for r in empty_renderers]
                    continue

                # Setup Camera
                obj = gt_obj_labels[obj_idx]
                half_h = obj.h / 2.0
                obj_mid_cen = obj.t - [0, half_h, 0]
                cam_dist = 4.0

                obj_z = obj.t[2]
                if obj_z < 10:
                    vtk_pc_depth.vtk_actor.GetProperty().SetPointSize(3)
                elif obj_z < 20:
                    vtk_pc_depth.vtk_actor.GetProperty().SetPointSize(4)
                elif obj_z < 30:
                    vtk_pc_depth.vtk_actor.GetProperty().SetPointSize(5)
                else:
                    vtk_pc_depth.vtk_actor.GetProperty().SetPointSize(6)

                obj_view_ang = obj_utils.get_viewing_angle_box_3d(obj.box_3d(), cam_p)

                for cam_pos_idx, cam_rot in enumerate(
                        np.linspace(np.deg2rad(-25), np.deg2rad(25), num_cam_pos)):

                    cam_rot = cam_rot - obj_view_ang

                    cam_x = cam_dist * np.sin(cam_rot)
                    cam_z = -cam_dist * np.cos(cam_rot)
                    cam_pos = obj_mid_cen + [cam_x, 0, cam_z]
                    cam_focal_point = obj_mid_cen

                    cam_vtk_renderer = all_vtk_renderers[obj_idx][cam_pos_idx]

                    cam_vtk_renderer.DrawOn()
                    current_cam = cam_vtk_renderer.GetActiveCamera()
                    current_cam.SetPosition(*cam_pos)
                    current_cam.SetFocalPoint(*cam_focal_point)

                    # Reset the clipping range to show all points
                    cam_vtk_renderer.ResetCameraClippingRange()

            render_time = time.time()
            window_to_image_filter.Update()
            print('render_time', time.time() - render_time)

            image_data_to_np_time = time.time()
            vtk_image = vtkImageData_to_np(window_to_image_filter.GetOutput())
            print('image_data_to_np_time', time.time() - image_data_to_np_time)
            print(vtk_image.shape)

            print('total_time', time.time() - total_time)

            # cv2.imshow('vtk_image', vtk_image)
            # cv2.waitKey(0)

            print('-----')


if __name__ == "__main__":
    main()
