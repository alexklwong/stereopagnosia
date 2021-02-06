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
import torch
import global_constants as settings


class StereoModel(object):
    '''
    Wrapper class for all stereo models

    Args:
        method : str
            stereo model to use
        device : torch.device
            device to run optimization
    '''

    def __init__(self,
                 method,
                 device=torch.device(settings.CUDA)):

        self.method = method

        if method == 'psmnet':
            from psmnet_model import PSMNetModel
            self.model = PSMNetModel(device=device)
        elif method == 'deeppruner':
            from deeppruner_model import DeepPrunerModel
            self.model = DeepPrunerModel(device=device)
        elif method == 'aanet':
            from aanet_model import AANetModel
            self.model = AANetModel(device=device)

    def forward(self, image0, image1=None):
        '''
        Forwards stereo pair through network

        Args:
            image0 : tensor
                N x C x H x W left image
            image1 : tensor
                N x C x H x W right image

        Returns:
            tensor : N x 1 x H x W disparity
        '''
        if image1 is None:
            image0, image1 = image0

        outputs = self.model.forward(image0, image1)

        return outputs

    def compute_loss(self, image0, image1, ground_truth):
        '''
        Computes training loss

        Args:
            image0 : tensor
                N x C x H x W left image
            image1 : tensor
                N x C x H x W right image
            ground_truth : tensor
                N x 1 x H x W disparity

        Returns:
            float : loss
        '''

        return self.model.compute_loss(image0, image1, ground_truth)

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        return self.model.parameters()

    def named_parameters(self):
        '''
        Returns the list of named parameters in the model

        Returns:
            list : list of parameters
        '''

        return self.model.named_parameters()

    def train(self):
        '''
        Sets model to training mode
        '''

        return self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        return self.model.eval()

    def save_model(self, save_path):
        '''
        Stores weights into a checkpoint
        Args:
            save_path : str
                path to model weights
        '''

        self.model.save_model(save_path)

    def restore_model(self, restore_path):
        '''
        Loads weights from checkpoint
        Args:
            restore_path : str
                path to model weights
        '''

        self.model.restore_model(restore_path)
