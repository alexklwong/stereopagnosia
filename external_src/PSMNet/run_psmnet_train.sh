export CUDA_VISIBLE_DEVICES=1,3

python finetune.py \
--maxdisp 192 \
--model stackhourglass \
--datatype 2015 \
--datapath training_directory_stereo_2015_256x640/ \
--epochs 300 \
--loadmodel ../../pretrained-models/PSMNet/pretrained_model_KITTI2015.tar \
--savemodel saved_model_256x640_input_diversity/
