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
from perturb_main import run


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--image0_path',
    type=str, required=True, help='Path to list of left image paths')
parser.add_argument('--image1_path',
    type=str, required=True, help='Path to list of right image paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth disparity paths')
# Run settings
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of each sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
# Perturb model settings
parser.add_argument('--perturb_method',
    type=str, default=settings.PERTURB_METHOD, help='Perturb method available: %s' % settings.PERTURB_METHOD_AVAILABLE)
parser.add_argument('--perturb_mode',
    type=str, default='both', help='Perturb modes available: %s' % settings.PERTURB_MODE_AVAILABLE)
parser.add_argument('--output_norm',
    type=float, default=settings.OUTPUT_NORM, help='Output norm of noise')
parser.add_argument('--n_step',
    type=int, default=settings.N_STEP, help='Number of steps to optimize perturbations')
parser.add_argument('--learning_rate',
    type=float, default=2e-3, help='Learning rate (alpha) to use for optimizing perturbations')
parser.add_argument('--momentum',
    type=float, default=0.47, help='Momentum (beta) used for momentum iterative fast gradient sign method')
parser.add_argument('--probability_diverse_input',
    type=float, default=0.00, help='Probability (p) to use diverse input')
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
        ground_truth_path=args.ground_truth_path,
        # Run settings
        n_height=args.n_height,
        n_width=args.n_width,
        # Perturbations Settings
        perturb_method=args.perturb_method,
        perturb_mode=args.perturb_mode,
        output_norm=args.output_norm,
        n_step=args.n_step,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        probability_diverse_input=args.probability_diverse_input,
        # Stereo model settings
        stereo_method=args.stereo_method,
        stereo_model_restore_path=args.stereo_model_restore_path,
        # Output settings
        output_path=args.output_path,
        # Hardware settings
        device=args.device)
