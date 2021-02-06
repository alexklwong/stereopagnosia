#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# pretrained_models/AANet/aanet_kitti12-e20bb24d.pth
# pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.02 \
--perturb_method ifgsm \
--perturb_mode both \
--n_step 40 \
--learning_rate 2e-3 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/aanet/ifgsm/both_norm2e2_lr2e3 \
--device gpu

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.01 \
--perturb_method ifgsm \
--perturb_mode both \
--n_step 40 \
--learning_rate 2.5e-4 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/aanet/ifgsm/both_norm1e2_lr25e4 \
--device gpu

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.005 \
--perturb_method ifgsm \
--perturb_mode both \
--n_step 40 \
--learning_rate 1.25e-4 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/aanet/ifgsm/both_norm5e3_lr125e4 \
--device gpu

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.002 \
--perturb_method ifgsm \
--perturb_mode both \
--n_step 40 \
--learning_rate 5e-5 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/aanet/ifgsm/both_norm2e3_lr5e5 \
--device gpu