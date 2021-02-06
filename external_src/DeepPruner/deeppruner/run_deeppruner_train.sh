export CUDA_VISIBLE_DEVICES=2,3

python finetune_kitti.py \
--loadmodel ../../../pretrained-models/DeepPruner/DeepPruner-best-kitti.tar \
--savemodel adverse_trained_models_256x640_input_diversity/ \
--train_datapath_2015 training_directory_stereo_2015_256x640 \
--val_datapath_2015 val_directory_stereo_2015_256x640 \
--logging_filename ./finetune_kitti_256x640_id.log
