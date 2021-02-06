#!/bin/bash

export CUDA_VISIBLE_DEVICES=2


# PSMNet -> AANet
python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm2e2/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm2e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/psmnet/fgsm/both_norm2e2/aanet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm1e2/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm1e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/psmnet/fgsm/both_norm1e2/aanet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm5e3/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm5e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/psmnet/fgsm/both_norm5e3/aanet \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm2e3/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm2e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method aanet \
--stereo_model_restore_path pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth \
--output_path perturb_models/psmnet/fgsm/both_norm2e3/aanet \
--device gpu

# PSMNet -> DeepPruner
python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm2e2/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm2e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method deeppruner \
--stereo_model_restore_path pretrained_models/DeepPruner/DeepPruner-best-kitti.tar \
--output_path perturb_models/psmnet/fgsm/both_norm2e2/deeppruner \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm1e2/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm1e2/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method deeppruner \
--stereo_model_restore_path pretrained_models/DeepPruner/DeepPruner-best-kitti.tar \
--output_path perturb_models/psmnet/fgsm/both_norm1e2/deeppruner \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm5e3/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm5e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method deeppruner \
--stereo_model_restore_path pretrained_models/DeepPruner/DeepPruner-best-kitti.tar \
--output_path perturb_models/psmnet/fgsm/both_norm5e3/deeppruner \
--device gpu

python src/run_transferability.py \
--image0_path testing/kitti_scene_flow_image0.txt \
--image1_path testing/kitti_scene_flow_image1.txt \
--noise0_dirpath perturb_models/psmnet/fgsm/both_norm2e3/noise0_output \
--noise1_dirpath perturb_models/psmnet/fgsm/both_norm2e3/noise1_output \
--ground_truth_path testing/kitti_scene_flow_test_disparity.txt \
--n_height 256 \
--n_width 640 \
--stereo_method deeppruner \
--stereo_model_restore_path pretrained_models/DeepPruner/DeepPruner-best-kitti.tar \
--output_path perturb_models/psmnet/fgsm/both_norm2e3/deeppruner \
--device gpu

