#!/bin/bash

export CUDA_VISIBLE_DEVICES=1


# DeepPruner -> AANet
python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm2e2/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm2e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/deeppruner/fgsm/both_norm2e2/aanet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm1e2/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm1e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/deeppruner/fgsm/both_norm1e2/aanet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm5e3/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm5e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/deeppruner/fgsm/both_norm5e3/aanet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm2e3/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm2e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/deeppruner/fgsm/both_norm2e3/aanet \
--device gpu

# DeepPruner -> PSMNet
python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm2e2/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm2e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/deeppruner/fgsm/both_norm2e2/psmnet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm1e2/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm1e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/deeppruner/fgsm/both_norm1e2/psmnet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm5e3/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm5e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/deeppruner/fgsm/both_norm5e3/psmnet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_test_image0.txt \
--image1_path testing/kitti_scene_flow_test_image1.txt \
--noise0_dirpath perturb_models/deeppruner/fgsm/both_norm2e3/noise0_output \
--noise1_dirpath perturb_models/deeppruner/fgsm/both_norm2e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_models/PSMNet/pretrained_model_KITTI2015.tar \
--output_path perturb_models/deeppruner/fgsm/both_norm2e3/psmnet \
--device gpu

