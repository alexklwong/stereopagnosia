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
import torch.utils.data
import data_utils


class StereoDataset(torch.utils.data.Dataset):
    '''
    Loads a stereo pair

    Args:
        image0_paths : str
            path to left image
        image1_paths : str
            path to right image
        ground_truth_paths : str
            path to disparity ground truth
        shape : list
            (H, W) to resize images
        normalize : bool
            if set, normalize images between 0 and 1
    '''

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 ground_truth_paths,
                 shape=(None, None),
                 normalize=True):

        self.image0_paths = image0_paths
        self.image1_paths = image1_paths
        self.ground_truth_paths = ground_truth_paths
        self.n_height = shape[0]
        self.n_width = shape[1]
        self.normalize = normalize

    def __getitem__(self, index):

        shape_resize = (self.n_height, self.n_width)

        # Load images
        image0 = data_utils.load_image(
            self.image0_paths[index],
            shape=shape_resize,
            normalize=self.normalize)

        image1 = data_utils.load_image(
            self.image1_paths[index],
            shape=shape_resize,
            normalize=self.normalize)

        # Load ground truth
        ground_truth = data_utils.load_disparity(
            self.ground_truth_paths[index],
            shape=shape_resize)

        return image0, image1, ground_truth

    def __len__(self):
        return len(self.image0_paths)
