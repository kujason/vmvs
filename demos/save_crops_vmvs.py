import os
import sys

import numpy as np
import vtk

import md3d
from md3d.builders.dataset_builder import DatasetBuilder
from md3d.datasets.kitti import obj_utils, calib_utils
from md3d.datasets.kitti.kitti_prop_dataset import KittiPropDataset
from md3d.datasets.kitti.obj_utils import Difficulty
from scene_vis.vtk_wrapper import vtk_utils
from scene_vis.vtk_wrapper.vtk_point_cloud import VtkPointCloud


def main(split_idx=None, num_splits=None):

    ##############################
    # Options
    ##############################

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

    classes = ['Pedestrian']
    difficulty = Difficulty.ALL

    num_cam_pos = 11

    output_dir = md3d.data_dir() + '/outputs/vmvs/{}_views'.format(num_cam_pos)
    os.makedirs(output_dir, exist_ok=True)

    ##############################
    # End of Options
    ##############################

    if split_idx is None:
        samples_to_use = dataset.sample_names
    else:
        samples_split = np.array_split(dataset.sample_names, num_splits)
        samples_to_use = samples_split[split_idx]

    area_extents = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]], dtype=np.float32)

    ##############################
    # Vtk Visualization
    ##############################
    vtk_pc_depth = VtkPointCloud()
    vtk_pc_depth.vtk_actor.GetProperty().SetPointSize(4)

    # Create Renderer
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)
    vtk_renderer.AddActor(vtk_pc_depth.vtk_actor)

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName("Obj Point Cloud")
    vtk_render_window.SetSize(224, 224)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    # Add custom interactor to toggle actor visibilities
    custom_interactor = vtk_utils.ToggleActorsInteractorStyle(
        [
            vtk_pc_depth.vtk_actor,
        ],
        vtk_renderer
    )

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Roll(180.0)

    vtk_render_window_interactor.SetInteractorStyle(custom_interactor)
    vtk_win_to_img_filter, vtk_png_writer = vtk_utils.setup_screenshots(vtk_render_window)

    for sample_idx, sample_name in enumerate(samples_to_use):

        # Get bounding boxes
        all_gt_obj_labels = obj_utils.read_labels(dataset.kitti_label_dir, sample_name)
        gt_obj_labels, _ = obj_utils.filter_labels(
            all_gt_obj_labels,
            classes=classes,
            difficulty=difficulty)

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

            vtk_pc_depth.set_points(depth_point_cloud.T, depth_point_colours)

            for obj_idx in range(len(gt_obj_labels)):
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

                    current_cam = vtk_renderer.GetActiveCamera()
                    current_cam.SetPosition(*cam_pos)
                    current_cam.SetFocalPoint(*cam_focal_point)

                    # Reset the clipping range to show all points
                    vtk_renderer.ResetCameraClippingRange()

                    # Render in VTK
                    vtk_render_window.Render()

                    output_path = output_dir + '/{}_{:02d}_{:02d}.png'.format(
                        sample_name, obj_idx, cam_pos_idx)
                    vtk_utils.save_screenshot(output_path, vtk_win_to_img_filter, vtk_png_writer)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        split_idx = int(sys.argv[1])
        num_splits = int(sys.argv[2])
        print('split / num_splits', split_idx, num_splits)
        main(split_idx, num_splits)
    else:
        main()
