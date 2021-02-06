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
import os, sys
import torch, torchvision
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join('external_src', 'DeepPruner'))
sys.path.insert(0, os.path.join('external_src', 'DeepPruner', 'deeppruner'))
sys.path.insert(0, os.path.join('external_src', 'DeepPruner', 'deeppruner', 'models'))
from external_src.DeepPruner.deeppruner.models.deeppruner import DeepPruner
from external_src.DeepPruner.deeppruner.loss_evaluation import loss_evaluation
import global_constants as settings


class DeepPrunerModel(object):
    '''
    Wrapper class for DeepPruner model

    Args:
        device : torch.device
            device to run optimization
    '''

    def __init__(self, device=torch.device(settings.CUDA)):

        self.device = device

        # Restore depth prediction network
        self.model = DeepPruner()

        # Move to device
        self.to(self.device)
        self.eval()

    def forward(self, image0, image1):
        '''
        Forwards stereo pair through the network

        Args:
            image0 : tensor
                N x C x H x W left image
            image1 : tensor
                N x C x H x W right image

        Returns:
            tensor : N x 1 x H x W disparity if mode is 'eval'
            list[tensor] : N x 1 x H x W disparity if mode is 'train'
        '''

        # Transform inputs
        image0, image1, \
            padding_top, padding_right = self.transform_inputs(image0, image1)

        # Forward through network
        outputs = self.model(image0, image1)

        if self.mode == 'eval':
            # Get finest output
            output = torch.unsqueeze(outputs[0], dim=1)

            # If we padded the input, then crop
            if padding_top != 0 or padding_right != 0:
                output = output[:, :, padding_top:, :-padding_right]

            return output

        elif self.mode == 'train':
            outputs = [
                torch.unsqueeze(output, dim=1) for output in outputs
            ]

            if padding_top != 0 or padding_right != 0:
                outputs = [
                    output[:, :, padding_top:, :-padding_right]
                    for output in outputs
                ]

            return outputs

    def transform_inputs(self, image0, image1):
        '''
        Transforms the stereo pair using standard normalization as a preprocessing step

        Args:
            image0 : tensor
                N x C x H x W left image
            image1 : tensor
                N x C x H x W right image

        Returns:
            tensor : N x 3 x H x W left image
            tensor : N x 3 x H x W right image
            int : padding applied to top of images
            int : padding applied to right of images
        '''

        # Dataset mean and standard deviations
        normal_mean_var = {
            'mean' : [0.485, 0.456, 0.406],
            'std' : [0.229, 0.224, 0.225]
        }

        transform_func = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(**normal_mean_var)])

        n_batch, _, n_height, n_width = image0.shape

        image0 = torch.chunk(image0, chunks=n_batch, dim=0)
        image1 = torch.chunk(image1, chunks=n_batch, dim=0)

        image0 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image0
        ], dim=0)
        image1 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image1
        ], dim=0)

        downsample_scale = 8.0 * settings.DEEPPRUNER_COST_AGGREGATOR_SCALE

        # Pad images along top and right dimensions
        padding_top = int(downsample_scale - (n_height % downsample_scale))
        padding_right = int(downsample_scale - (n_width % downsample_scale))

        image0 = torch.nn.functional.pad(
            image0,
            (0, padding_right, padding_top, 0, 0, 0),
            mode='constant',
            value=0)
        image1 = torch.nn.functional.pad(
            image1,
            (0, padding_right, padding_top, 0, 0, 0),
            mode='constant',
            value=0)

        return image0, image1, padding_top, padding_right

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

        # Switch to training mode
        self.model.mode = 'training'
        self.mode = 'train'

        mask = ground_truth > 0
        mask.detach_()

        result = self.forward(image0, image1)

        loss, _ = loss_evaluation(
            result,
            ground_truth,
            mask,
            settings.DEEPPRUNER_COST_AGGREGATOR_SCALE)

        # Switch back to evaluation mode
        self.model.mode = 'evaluation'
        self.mode = 'eval'

        return loss

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

        self.model.train()
        self.model.mode = 'training'
        self.mode = 'train'

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()
        self.model.mode = 'evaluation'
        self.mode = 'eval'

    def to(self, device):
        '''
        Moves model to device
        Args:
            device : Torch device
                CPU or GPU/CUDA device
        '''

        # Move to device
        self.model.to(device)

    def save_model(self, save_path):
        '''
        Stores weights into a checkpoint
        Args:
            save_path : str
                path to model weights
        '''

        checkpoint = {
            'state_dict' : self.model.state_dict()
        }

        torch.save(checkpoint, save_path)

    def restore_model(self, restore_path):
        '''
        Loads weights from a checkpoint
        Args:
            restore_path : str
                path to model weights
        '''

        self.model = torch.nn.DataParallel(self.model)
        state_dict = torch.load(restore_path)
        self.model.load_state_dict(state_dict['state_dict'], strict=True)
