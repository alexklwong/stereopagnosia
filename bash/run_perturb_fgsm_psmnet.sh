#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# pretrained_models/PSMNet/pretrained_model_KITTI2012.tar
# pretrained_models/PSMNet/pretrained_model_KITTI2015.tar
# external_src/PSMNet/saved_model_256x640_input_diversity/finetune_100.tar

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.02 \
--perturb_method fgsm \
--perturb_mode both \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/psmnet/fgsm/both_norm2e2 \
--device gpu

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.01 \
--perturb_method fgsm \
--perturb_mode both \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/psmnet/fgsm/both_norm1e2 \
--device gpu

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.005 \
--perturb_method fgsm \
--perturb_mode both \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/psmnet/fgsm/both_norm5e3 \
--device gpu

python src/run_perturb_model.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--output_norm 0.002 \
--perturb_method fgsm \
--perturb_mode both \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/psmnet/fgsm/both_norm2e3 \
--device gpu
