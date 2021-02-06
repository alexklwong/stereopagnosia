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
import numpy as np
from PIL import Image


def read_paths(filepath):
    '''
    Reads line delimited paths from file

    Args:
        filepath : str
            path to file containing line delimited paths
    Returns:
        list : list of paths
    '''
    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            # If there was nothing to read
            if path == '':
                break
            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Args:
        filepath : str
            path to file to save paths
        paths : list
            paths to write into file
    '''
    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, shape, normalize=True):
    '''
    Loads an RGB image

    Args:
        path : str
            path to image
        shape : list
            (H, W) of image
        normalize : bool
            if set, normalize image between 0 and 1

    Returns:
        numpy : 3 x H x W RGB image
    '''
    n_height, n_width = shape

    # Load image and resize
    image = Image.open(path)

    if n_height is not None and n_width is not None:
        image = image.resize((n_width, n_height), Image.LANCZOS)

    image = np.asarray(image, np.float32)
    image = np.transpose(image, (2, 0, 1))

    # Normalize
    image = image / 255.0 if normalize else image

    return image

def load_disparity(path, shape=(None, None)):
    '''
    Loads a disparity image

    Args:
        path : str
            path to disparity image
        shape : list
            (H, W) of disparity image
    Returns:
        numpy : H x W disparity image
    '''

    # Load image and resize
    disparity = Image.open(path).convert('I')
    o_width, o_height = disparity.size

    n_height, n_width = shape

    if n_height is None or n_width is None:
        n_height = o_height
        n_width = o_width

    # Resize to dataset shape
    disparity = disparity.resize((n_width, n_height), Image.NEAREST)

    # Convert unsigned int16 to disparity values
    disparity = np.asarray(disparity, np.uint16)
    disparity = disparity / 256.0

    # Adjust disparity based on resize
    scale = np.asarray(n_width, np.float32) / np.asarray(o_width, np.float32)
    disparity = disparity * scale

    return np.asarray(disparity, np.float32)
