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

'''
Downloading models
'''

GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

DEEPPRUNER_KITTI_BEST_STEREO_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1mSendpKq0vdQMr5XVp5zAHm37dwEgcun')

DEEPPRUNER_KITTI_FAST_STEREO_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1jJAawAUcTpmhj8YdG7BWonglKpfQ8oMN')

PRETRAINED_MODELS_DIRPATH = 'pretrained_models'
DEEPPRUNER_MODELS_DIRNAME = 'DeepPruner'

DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILENAME = 'DeepPruner-best-kitti.tar'
DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILENAME = 'DeepPruner-fast-kitti.tar'

DEEPPRUNER_MODELS_DIRPATH = os.path.join(PRETRAINED_MODELS_DIRPATH, DEEPPRUNER_MODELS_DIRNAME)

DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILEPATH = os.path.join(
    PRETRAINED_MODELS_DIRPATH, DEEPPRUNER_MODELS_DIRNAME, DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILENAME)
DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILEPATH = os.path.join(
    PRETRAINED_MODELS_DIRPATH, DEEPPRUNER_MODELS_DIRNAME, DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILENAME)


if not os.path.exists(DEEPPRUNER_MODELS_DIRPATH):
    os.makedirs(DEEPPRUNER_MODELS_DIRPATH)

if not os.path.exists(DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILEPATH):
    print('Downloading {} to {}'.format(
        DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILENAME, DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILEPATH))
    gdown.download(DEEPPRUNER_KITTI_BEST_STEREO_MODEL_URL, DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILEPATH, quiet=False)
else:
    print('Found {} at {}'.format(
        DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILENAME, DEEPPRUNER_KITTI_BEST_STEREO_MODEL_FILEPATH))

if not os.path.exists(DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILEPATH):
    print('Downloading {} to {}'.format(
        DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILENAME, DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILEPATH))
    gdown.download(DEEPPRUNER_KITTI_FAST_STEREO_MODEL_URL, DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILEPATH, quiet=False)
else:
    print('Found {} at {}'.format(
        DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILENAME, DEEPPRUNER_KITTI_FAST_STEREO_MODEL_FILEPATH))
