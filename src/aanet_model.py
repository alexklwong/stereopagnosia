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
sys.path.insert(0, os.path.join('external_src', 'aanet'))
sys.path.insert(0, os.path.join('external_src', 'aanet', 'dataloader'))
sys.path.insert(0, os.path.join('external_src', 'aanet', 'nets'))
from external_src.aanet.nets import AANet
from external_src.aanet.utils import utils
import global_constants as settings


class AANetModel(object):
    '''
    Wrapper class for AANet model

    Args:
        device : torch.device
            device to run optimization
    '''

    def __init__(self, device=torch.device(settings.CUDA)):

        self.device = device

        # Restore depth prediction network
        self.model = AANet(
            settings.MAX_DISPARITY_AANET,
            num_downsample=2,
            feature_type='aanet',
            no_feature_mdconv=False,
            feature_pyramid=False,
            feature_pyramid_network=True,
            feature_similarity='correlation',
            aggregation_type='adaptive',
            num_scales=3,
            num_fusions=6,
            num_stage_blocks=1,
            num_deform_blocks=3,
            no_intermediate_supervision=True,
            refinement_type='stereodrnet',
            mdconv_dilation=2,
            deformable_groups=2)

        # Move to device
        self.to(self.device)
        self.eval()

    def forward(self, image0, image1=None):
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
        if image1 is None:
            image0, image1 = image0

        # Transform inputs
        image0, image1, \
            padding_top, padding_right = self.transform_inputs(image0, image1)

        # Forward through network (make sure )
        outputs = self.model(image0, image1)

        if self.mode == 'eval':
            # Get finest output
            output = torch.unsqueeze(outputs[-1], dim=1)

            # If we padded the input, then crop
            if padding_top != 0 or padding_right != 0:
                output = output[:, :, padding_top:, :-padding_right]

            return output

        elif self.mode == 'train':
            outputs = [
                torch.unsqueeze(output, dim=1) for output in outputs
            ]

            # If we padded the input, then crop
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
        '''

        normal_mean_var = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
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

        downsample_scale = 12

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
                N x C x H x W left RGB image
            image1 : tensor
                N x C x H x W right RGB image
            ground_truth : tensor
                N x 1 x H x W  disparity

        Returns:
            float : loss
        '''

        mask_ground_truth = \
            (ground_truth > 0) & (ground_truth < settings.MAX_DISPARITY_AANET)

        mask_ground_truth.detach_()

        # Switch to training mode
        self.mode = 'train'

        outputs = self.forward(image0, image1)
        output = outputs[-1]

        # Select outputs where disparity is defined
        loss = torch.nn.functional.smooth_l1_loss(
            output[mask_ground_truth],
            ground_truth[mask_ground_truth],
            reduction='mean')

        # Switch back to evaluation mode
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
        self.mode = 'train'

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()
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
        Loads weights from checkpoint

        Args:
            restore_path : str
                path to model weights
        '''

        utils.load_pretrained_net(self.model, restore_path, no_strict=True)
        self.model = torch.nn.DataParallel(self.model)
