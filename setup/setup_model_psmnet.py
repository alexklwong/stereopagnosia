'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Mukund Mundhra <mukundmundhra@cs.ucla.edu>

If this code is useful to you, please consider citing the following paper:

A. Wong, M. Mundhra, and S. Soatto. Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.
https://arxiv.org/pdf/2009.10142.pdf

@inproceedings{wong2021stereopagnosia,
  title={Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations},
  author={Wong, Alex and Mundhra, Mukund and Soatto, Stefano},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  year={2021}
}
'''
import os, gdown


TRAIN_REFS_DIRPATH = 'training'
TEST_REFS_DIRPATH = 'testing'

GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

PSMNET_KITTI_2012_STEREO_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ')
PSMNET_KITTI_2015_STEREO_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9')

PRETRAINED_MODELS_DIRPATH = 'pretrained_models'
PSMNET_MODELS_DIRNAME = 'PSMNet'

PSMNET_KITTI_2012_STEREO_MODEL_FILENAME = 'pretrained_model_KITTI2012.tar'
PSMNET_KITTI_2015_STEREO_MODEL_FILENAME = 'pretrained_model_KITTI2015.tar'

PSMNET_MODELS_DIRPATH = os.path.join(PRETRAINED_MODELS_DIRPATH, PSMNET_MODELS_DIRNAME)

PSMNET_KITTI_2012_STEREO_MODEL_FILEPATH = os.path.join(
    PSMNET_MODELS_DIRPATH, PSMNET_KITTI_2012_STEREO_MODEL_FILENAME)

PSMNET_KITTI_2015_STEREO_MODEL_FILEPATH = os.path.join(
    PSMNET_MODELS_DIRPATH, PSMNET_KITTI_2015_STEREO_MODEL_FILENAME)


if not os.path.exists(PSMNET_MODELS_DIRPATH):
    os.makedirs(PSMNET_MODELS_DIRPATH)

if not os.path.exists(PSMNET_KITTI_2012_STEREO_MODEL_FILEPATH):
    print('Downloading {} to {}'.format(
        PSMNET_KITTI_2012_STEREO_MODEL_FILENAME, PSMNET_KITTI_2012_STEREO_MODEL_FILEPATH))
    gdown.download(PSMNET_KITTI_2012_STEREO_MODEL_URL, PSMNET_KITTI_2012_STEREO_MODEL_FILEPATH, quiet=False)
else:
    print('Found {} at {}'.format(
        PSMNET_KITTI_2012_STEREO_MODEL_FILENAME, PSMNET_KITTI_2012_STEREO_MODEL_FILEPATH))

if not os.path.exists(PSMNET_KITTI_2015_STEREO_MODEL_FILEPATH):
    print('Downloading {} to {}'.format(
        PSMNET_KITTI_2015_STEREO_MODEL_FILENAME, PSMNET_KITTI_2015_STEREO_MODEL_FILEPATH))
    gdown.download(PSMNET_KITTI_2015_STEREO_MODEL_URL, PSMNET_KITTI_2015_STEREO_MODEL_FILEPATH, quiet=False)
else:
    print('Found {} at {}'.format(
        PSMNET_KITTI_2015_STEREO_MODEL_FILENAME, PSMNET_KITTI_2015_STEREO_MODEL_FILEPATH))
