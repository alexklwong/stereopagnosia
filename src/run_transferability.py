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
import argparse
import global_constants as settings
from transferability import run


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--image0_path',
    type=str, required=True, help='Path to list of left image paths')
parser.add_argument('--image1_path',
    type=str, required=True, help='Path to list of right image paths')
parser.add_argument('--noise0_dirpath',
    type=str, required=True, help='Path to left noise directory')
parser.add_argument('--noise1_dirpath',
    type=str, required=True, help='Path to right noise directory')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth disparity paths')
# Run settings
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
# Stereo model settings
parser.add_argument('--stereo_method',
    type=str, default='psmnet', help='Stereo method available: %s' % settings.STEREO_METHOD_AVAILABLE)
parser.add_argument('--stereo_model_restore_path',
    type=str, default='', help='Path to restore model checkpoint')
# Output settings
parser.add_argument('--output_path',
    type=str, default=settings.OUTPUT_PATH, help='Path to save outputs')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    args.stereo_method = args.stereo_method.lower()

    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    run(image0_path=args.image0_path,
        image1_path=args.image1_path,
        noise0_dirpath=args.noise0_dirpath,
        noise1_dirpath=args.noise1_dirpath,
        ground_truth_path=args.ground_truth_path,
        # Run settings
        n_height=args.n_height,
        n_width=args.n_width,
        # Stereo model settings
        stereo_method=args.stereo_method,
        stereo_model_restore_path=args.stereo_model_restore_path,
        # Output settings
        output_path=args.output_path,
        # Hardware settings
        device=args.device)
