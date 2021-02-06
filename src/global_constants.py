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
import os


# Batch settings
N_BATCH                             = 8
N_HEIGHT                            = 256
N_WIDTH                             = 640
N_CHANNEL                           = 3

# Perturb method settings
PERTURB_METHOD                      = 'fgsm'
PERTURB_MODE                        = 'both'
OUTPUT_NORM                         = 0.02
N_STEP                              = 40
LEARNING_RATES                      = 2e-3
MOMENTUM                            = 0.47
PROBABILITY_DIVERSE_INPUT           = 0.00

PERTURB_METHOD_AVAILABLE            = ['fgsm',
                                       'ifgsm',
                                       'mifgsm',
                                       'gaussian',
                                       'uniform']
PERTURB_MODE_AVAILABLE              = ['both', 'left', 'right']

# Stereo method settings
STEREO_METHOD                       = 'psmnet'
STEREO_MODEL_RESTORE_PATH           = os.path.join('pretrained_models', 'PSMNet', 'pretrained_model_KITTI2015.tar')

MAX_DISPARITY_PSMNET                = 192
MAX_DISPARITY_AANET                 = 192
DEEPPRUNER_COST_AGGREGATOR_SCALE    = 4

# Output settings
OUTPUT_PATH                         = os.path.join('perturb_models', 'psmnet', 'fgsm', 'both_norm2e2')

# Hardware settings
DEVICE                          = 'cuda'
CUDA                            = 'cuda'
CPU                             = 'cpu'
GPU                             = 'gpu'
N_THREAD                        = 8

# Other Settings
RANDOM_SEED                     = 1
