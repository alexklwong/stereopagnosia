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
import os, sys, gdown, zipfile, glob
sys.path.insert(0, 'src')
import data_utils


TRAIN_REFS_DIRPATH = 'training'
TEST_REFS_DIRPATH = 'testing'


'''
Downloading models
'''
GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

AANET_KITTI_2012_STEREO_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1G2M6w0RIe6kZU_iHpOyrYg_cVd1Jc34Q')
AANET_KITTI_2015_STEREO_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1Qa1rSWQDBcW4D6LvvjDztfTLNGA_rgGn')

PRETRAINED_MODELS_DIRPATH = 'pretrained_models'
AANET_MODELS_DIRNAME = 'AANet'

AANET_KITTI_2012_STEREO_MODEL_FILENAME = 'aanet_kitti12-e20bb24d.pth'
AANET_KITTI_2015_STEREO_MODEL_FILENAME = 'aanet_kitti15-fb2a0d23.pth'

AANET_MODELS_DIRPATH = os.path.join(PRETRAINED_MODELS_DIRPATH, AANET_MODELS_DIRNAME)

AANET_KITTI_2012_STEREO_MODEL_FILEPATH = os.path.join(
    AANET_MODELS_DIRPATH, AANET_KITTI_2012_STEREO_MODEL_FILENAME)

AANET_KITTI_2015_STEREO_MODEL_FILEPATH = os.path.join(
    AANET_MODELS_DIRPATH, AANET_KITTI_2015_STEREO_MODEL_FILENAME)


'''
Pseudo groundtruth disparity
'''

# KITTI 2012 dataset (stereo flow)
STEREO_FLOW_EXTRAS_DISPARITY_URL = GOOGLE_DRIVE_BASE_URL.format('1ZJhraqgY1sL4UfHBrVojttCbvNAXfdj0')

STEREO_FLOW_ROOT_DIRPATH = os.path.join('data', 'kitti_stereo_flow', 'training')

STEREO_FLOW_EXTRAS_ROOT_DIRPATH = os.path.join(
    'data', 'kitti_stereo_flow_extras', 'training')

STEREO_FLOW_EXTRAS_DISPARITY_DIRPATH = os.path.join(
    STEREO_FLOW_EXTRAS_ROOT_DIRPATH, 'disp_occ_pseudo_gt')

STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH = os.path.join(
    STEREO_FLOW_EXTRAS_ROOT_DIRPATH, 'kitti_2012_disp_occ_pseudo_gt.zip')

STEREO_FLOW_IMAGE0_DIRPATH = os.path.join(STEREO_FLOW_ROOT_DIRPATH, 'colored_0')

STEREO_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_stereo_flow_all_disparity_pseudo.txt')
STEREO_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_stereo_flow_train_disparity_pseudo.txt')
STEREO_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_stereo_flow_test_disparity_pseudo.txt')


# KITTI 2015 dataset (scene flow)
SCENE_FLOW_EXTRAS_DISPARITY_URL = GOOGLE_DRIVE_BASE_URL.format('14NGQp9CwIVNAK8ZQ6GSNeGraFGtVGOce')

SCENE_FLOW_ROOT_DIRPATH = os.path.join('data', 'kitti_scene_flow', 'training')

SCENE_FLOW_EXTRAS_ROOT_DIRPATH = os.path.join(
    'data', 'kitti_scene_flow_extras', 'training')

SCENE_FLOW_EXTRAS_DISPARITY_DIRPATH = os.path.join(
    SCENE_FLOW_EXTRAS_ROOT_DIRPATH, 'disp_occ_0_pseudo_gt')

SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH = os.path.join(
    SCENE_FLOW_EXTRAS_ROOT_DIRPATH, 'kitti_2015_disp_occ_0_pseudo_gt.zip')

SCENE_FLOW_IMAGE0_DIRPATH = os.path.join(SCENE_FLOW_ROOT_DIRPATH, 'image_2')

SCENE_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_scene_flow_all_disparity_pseudo.txt')
SCENE_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_scene_flow_train_disparity_pseudo.txt')
SCENE_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_scene_flow_test_disparity_pseudo.txt')


'''
Download pretrained AANet models
'''

if not os.path.exists(AANET_MODELS_DIRPATH):
    os.makedirs(AANET_MODELS_DIRPATH)

if not os.path.exists(AANET_KITTI_2012_STEREO_MODEL_FILEPATH):
    print('Downloading {} to {}'.format(
        AANET_KITTI_2012_STEREO_MODEL_FILENAME, AANET_KITTI_2012_STEREO_MODEL_FILEPATH))
    gdown.download(AANET_KITTI_2012_STEREO_MODEL_URL, AANET_KITTI_2012_STEREO_MODEL_FILEPATH, quiet=False)
else:
    print('Found {} at {}'.format(
        AANET_KITTI_2012_STEREO_MODEL_FILENAME, AANET_KITTI_2012_STEREO_MODEL_FILEPATH))

if not os.path.exists(AANET_KITTI_2015_STEREO_MODEL_FILEPATH):
    print('Downloading {} to {}'.format(
        AANET_KITTI_2015_STEREO_MODEL_FILENAME, AANET_KITTI_2015_STEREO_MODEL_FILEPATH))
    gdown.download(AANET_KITTI_2015_STEREO_MODEL_URL, AANET_KITTI_2015_STEREO_MODEL_FILEPATH, quiet=False)
else:
    print('Found {} at {}'.format(
        AANET_KITTI_2015_STEREO_MODEL_FILENAME, AANET_KITTI_2015_STEREO_MODEL_FILEPATH))

'''
Download pseudo groundtruth disparity from GANet
'''

# KITTI 2012 dataset (stereo flow)
if not os.path.exists(STEREO_FLOW_EXTRAS_ROOT_DIRPATH):
    os.makedirs(STEREO_FLOW_EXTRAS_ROOT_DIRPATH)

if not os.path.exists(STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH):
    print('Downloading stereo flow pseudo groundtruth disparity to {}'.format(
        STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH))
    gdown.download(STEREO_FLOW_EXTRAS_DISPARITY_URL, STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH, quiet=False)
else:
    print('Found stereo flow pseudo groundtruthd disparity at {}'.format(
        STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH))

with zipfile.ZipFile(STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH, 'r') as z:
    z.extractall(STEREO_FLOW_EXTRAS_ROOT_DIRPATH)

stereo_flow_image0_paths = sorted(glob.glob(os.path.join(STEREO_FLOW_IMAGE0_DIRPATH, '*_10.png')))
stereo_flow_extras_disparity_paths = sorted(glob.glob(os.path.join(STEREO_FLOW_EXTRAS_DISPARITY_DIRPATH, '*_10.png')))

assert len(stereo_flow_image0_paths) == len(stereo_flow_extras_disparity_paths)

stereo_flow_extras_train_disparity_paths = stereo_flow_extras_disparity_paths[0:160]
stereo_flow_extras_test_disparity_paths = stereo_flow_extras_disparity_paths[160:]

# Write all paths to disk
print('Storing all {} stereo flow pseduo ground truth disparity file paths into: {}'.format(
    len(stereo_flow_extras_disparity_paths),
    STEREO_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH))
data_utils.write_paths(
    STEREO_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH,
    stereo_flow_extras_disparity_paths)

# Write training paths to disk
print('Storing {} training stereo flow pseudo ground truth disparity file paths into: {}'.format(
    len(stereo_flow_extras_train_disparity_paths),
    STEREO_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH))
data_utils.write_paths(
    STEREO_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH,
    stereo_flow_extras_train_disparity_paths)

# Write testing paths to disk
print('Storing {} testing pseudo stereo flow ground truth disparity file paths into: {}'.format(
    len(stereo_flow_extras_test_disparity_paths),
    STEREO_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH))
data_utils.write_paths(
    STEREO_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH,
    stereo_flow_extras_test_disparity_paths)


# KITTI 2015 dataset (scene flow)
if not os.path.exists(SCENE_FLOW_EXTRAS_ROOT_DIRPATH):
    os.makedirs(SCENE_FLOW_EXTRAS_ROOT_DIRPATH)

if not os.path.exists(SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH):
    print('Downloading scene flow pseudo groundtruth disparity to {}'.format(
        SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH))
    gdown.download(SCENE_FLOW_EXTRAS_DISPARITY_URL, SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH, quiet=False)
else:
    print('Found scene flow pseudo groundtruthd disparity at {}'.format(
        SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH))

with zipfile.ZipFile(SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH, 'r') as z:
    z.extractall(SCENE_FLOW_EXTRAS_ROOT_DIRPATH)

scene_flow_image0_paths = sorted(glob.glob(os.path.join(SCENE_FLOW_IMAGE0_DIRPATH, '*_10.png')))
scene_flow_extras_disparity_paths = sorted(glob.glob(os.path.join(SCENE_FLOW_EXTRAS_DISPARITY_DIRPATH, '*_10.png')))

assert len(scene_flow_image0_paths) == len(scene_flow_extras_disparity_paths)

scene_flow_extras_train_disparity_paths = scene_flow_extras_disparity_paths[0:160]
scene_flow_extras_test_disparity_paths = scene_flow_extras_disparity_paths[160:]

# Write all paths to disk
print('Storing all {} scene flow pseduo ground truth disparity file paths into: {}'.format(
    len(scene_flow_extras_disparity_paths),
    SCENE_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH))
data_utils.write_paths(
    SCENE_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH,
    scene_flow_extras_disparity_paths)

# Write training paths to disk
print('Storing {} training scene flow pseudo ground truth disparity file paths into: {}'.format(
    len(scene_flow_extras_train_disparity_paths),
    SCENE_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH))
data_utils.write_paths(
    SCENE_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH,
    scene_flow_extras_train_disparity_paths)

# Write testing paths to disk
print('Storing {} testing scene flow pseudo ground truth disparity file paths into: {}'.format(
    len(scene_flow_extras_test_disparity_paths),
    SCENE_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH))
data_utils.write_paths(
    SCENE_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH,
    scene_flow_extras_test_disparity_paths)
