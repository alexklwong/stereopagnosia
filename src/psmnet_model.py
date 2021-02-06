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
sys.path.insert(0, os.path.join('external_src', 'psmnet'))
sys.path.insert(0, os.path.join('external_src', 'psmnet', 'models'))
from external_src.PSMNet.models import stackhourglass
import global_constants as settings


class PSMNetModel(object):
    '''
    Wrapper class for PSMNet model

    Args:
        device : torch.device
            device to run optimization
    '''

    def __init__(self, device=torch.device(settings.CUDA)):

        self.device = device

        # Restore depth prediction network
        self.model = stackhourglass(settings.MAX_DISPARITY_PSMNET)

        # Move to device
        self.to(self.device)
        self.eval()
        self.model.eval()

    def forward(self, image0, image1=None):
        '''
        Forwards stereo pair through the network

        Args:
            image0 : tensor
                N x C x H x W left image
            image1 : tensor
                N x C x H x W right image

        Returns:
            tensor : N x H x W disparity if mode is 'eval'
            list[tensor] : N x H x W disparity if mode is 'train'
        '''
        if image1 is None:
            image0, image1 = image0

        # Transform inputs
        image0, image1, \
            padding_top, padding_right = self.transform_inputs(image0, image1)

        outputs = self.model(image0, image1)

        if self.mode == 'eval':
            # Get finest output
            output3 = torch.unsqueeze(outputs[2], dim=1)

            # If we padded the input, then crop
            if padding_top != 0 or padding_right != 0:
                output3 = output3[..., padding_top:, :-padding_right]

            return output3

        elif self.mode == 'train':
            outputs = [
                torch.unsqueeze(output, dim=1) for output in outputs
            ]

            output1, output2, output3 = outputs

            # If we padded the input, then crop
            if padding_top != 0 or padding_right != 0:
                output1 = output1[..., padding_top:, :-padding_right]
                output2 = output2[..., padding_top:, :-padding_right]
                output3 = output3[..., padding_top:, :-padding_right]

            return output1, output2, output3

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

        # Pad to width and height such that it is divisible by 16
        if n_height % 16 != 0:
            times = n_height // 16
            padding_top = (times + 1) * 16 - n_height
        else:
            padding_top = 0

        if n_width % 16 != 0:
            times = n_width // 16
            padding_right = (times + 1) * 16 - n_width
        else:
            padding_right = 0

        # Pad the images and expand at 0-th dimension to get batch
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
        self.mode = 'train'

        mask = ground_truth > 0
        mask.detach_()

        # Select ground truth where disparity is defined
        ground_truth = ground_truth[mask]

        output1, output2, output3 = self.forward(image0, image1)

        # Select outputs where disparity is defined
        output1 = output1[mask]
        output2 = output2[mask]
        output3 = output3[mask]

        loss1 = torch.nn.functional.smooth_l1_loss(output1, ground_truth, size_average=True)
        loss2 = torch.nn.functional.smooth_l1_loss(output2, ground_truth, size_average=True)
        loss3 = torch.nn.functional.smooth_l1_loss(output3, ground_truth, size_average=True)

        loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

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

        self.model = torch.nn.DataParallel(self.model)
        state_dict = torch.load(restore_path)
        self.model.load_state_dict(state_dict['state_dict'])
