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
import os, warnings, glob
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import torch
import datasets, data_utils
import global_constants as settings
from log_utils import log
from stereo_model import StereoModel
from perturb_main import validate


def run(image0_path,
        image1_path,
        noise0_dirpath,
        noise1_dirpath,
        ground_truth_path,
        # Run settings
        n_height=settings.N_HEIGHT,
        n_width=settings.N_WIDTH,
        # Stereo model settings
        stereo_method=settings.STEREO_METHOD,
        stereo_model_restore_path=settings.STEREO_MODEL_RESTORE_PATH,
        # Output settings
        output_path=settings.OUTPUT_PATH,
        # Hardware settings
        device=settings.DEVICE):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set up output and logging paths
    log_path = os.path.join(output_path, 'results.txt')

    # Set random seed
    torch.manual_seed(settings.RANDOM_SEED)
    np.random.seed(settings.RANDOM_SEED)

    if device.type == settings.CUDA:
        torch.cuda.manual_seed(settings.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

    # Read input paths
    image0_paths = data_utils.read_paths(image0_path)
    image1_paths = data_utils.read_paths(image1_path)
    ground_truth_paths = data_utils.read_paths(ground_truth_path)

    # Fetch paths to noise files in directory
    noise0_paths = sorted(glob.glob(os.path.join(noise0_dirpath, '*.npy')))
    noise1_paths = sorted(glob.glob(os.path.join(noise1_dirpath, '*.npy')))

    n_sample = len(image0_paths)

    assert n_sample == len(image1_paths)
    assert n_sample == len(ground_truth_paths)
    assert n_sample == len(noise0_paths)
    assert n_sample == len(noise1_paths)

    stereo_model = StereoModel(method=stereo_method, device=device)

    # Restore stereo model
    stereo_model.restore_model(stereo_model_restore_path)

    dataloader = torch.utils.data.DataLoader(
        datasets.StereoDataset(
            image0_paths,
            image1_paths,
            ground_truth_paths,
            shape=(n_height, n_width)),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    log('Run settings:', log_path)
    log('n_height=%d  n_width=%d' %
        (n_height, n_width),
        log_path)

    log('Input noise paths', log_path)
    log('noise0_dirpath=%s' %
        (noise0_dirpath),
        log_path)
    log('noise1_dirpath=%s' %
        (noise1_dirpath),
        log_path)

    log('Stereo model settings:', log_path)
    log('stereo_method=%s' %
        (stereo_method),
        log_path)
    log('stereo_model_restore_path=%s' %
        (stereo_model_restore_path),
        log_path)

    log('Output settings:', log_path)
    log('output_path=%s' %
        (output_path),
        log_path)

    log('Running...', log_path)

    disparities_origin = []
    noises0_output = []
    noises1_output = []
    images0_output = []
    images1_output = []
    disparities_output = []
    ground_truths = []

    for idx, (image0, image1, ground_truth) in enumerate(dataloader):
        # Load perturbations
        noise0 = np.load(noise0_paths[idx])
        noise1 = np.load(noise1_paths[idx])

        noises0_output.append(noise0)
        noises1_output.append(noise1)

        noise0 = torch.from_numpy(np.transpose(noise0, (2, 0, 1)))
        noise1 = torch.from_numpy(np.transpose(noise1, (2, 0, 1)))

        if device.type == settings.CUDA:
            image0 = image0.cuda()
            image1 = image1.cuda()
            ground_truth = ground_truth.cuda()
            noise0 = noise0.cuda()
            noise1 = noise1.cuda()

        if len(image0.shape) == 4 and image0.shape[0] == 1:
            noise0 = torch.unsqueeze(noise0, dim=0)
            noise1 = torch.unsqueeze(noise1, dim=0)
            ground_truth = torch.unsqueeze(ground_truth, dim=0)

        if noise0.shape[2:4] != image0.shape[2:4]:
            noise0 = torch.nn.functional.interpolate(
                noise0,
                size=image0.shape[2:4],
                mode='bilinear')
            noise1 = torch.nn.functional.interpolate(
                noise1,
                size=image0.shape[2:4],
                mode='bilinear')

        # Forward through through stereo network
        disparity_origin = stereo_model.forward(image0, image1)

        # Save original disparity
        disparities_origin.append(
            np.squeeze(disparity_origin.detach().cpu().numpy()))

        ground_truths.append(
            np.squeeze(ground_truth.cpu().numpy()))

        # Perturb images
        image0_output = image0 + noise0
        image1_output = image1 + noise1

        # Forward through network again
        disparity_output = stereo_model.forward(image0_output, image1_output)

        # Measure L1 loss from ground truth
        loss_func = torch.nn.L1Loss()
        loss = loss_func(disparity_output, ground_truth)

        # Save outputs
        images0_output.append(
            np.transpose(np.squeeze(image0_output.detach().cpu().numpy()), (1, 2, 0)))
        images1_output.append(
            np.transpose(np.squeeze(image1_output.detach().cpu().numpy()), (1, 2, 0)))
        disparities_output.append(
            np.squeeze(disparity_output.detach().cpu().numpy()))

        log('Sample={:3}/{:3}  L1 Loss={:.5f}'.format(
            idx + 1, n_sample, loss.item()),
            log_path)

        # Clean up
        del image0, image1
        del image0_output, image1_output
        del noise0, noise1
        del disparity_origin, disparity_output
        del ground_truth
        del loss

        if device.type == settings.CUDA:
            torch.cuda.empty_cache()

    # Perform validation
    with torch.no_grad():
        validate(
            noises0_output=noises0_output,
            noises1_output=noises1_output,
            disparities_output=disparities_output,
            ground_truths=ground_truths,
            device=device,
            log_path=log_path)

    log('Storing image and depth outputs into {}'.format(output_path), log_path)

    image0_output_path = os.path.join(output_path, 'image0_output')
    image1_output_path = os.path.join(output_path, 'image1_output')

    noise0_output_path = os.path.join(output_path, 'noise0_output')
    noise1_output_path = os.path.join(output_path, 'noise1_output')

    disparity_origin_path = os.path.join(output_path, 'disparity_origin')
    disparity_output_path = os.path.join(output_path, 'disparity_output')

    ground_truth_path = os.path.join(output_path, 'ground_truth')

    output_paths = [
        output_path,
        image0_output_path,
        image1_output_path,
        noise0_output_path,
        noise1_output_path,
        disparity_origin_path,
        disparity_output_path,
        ground_truth_path
    ]

    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    outputs = zip(
        images0_output,
        images1_output,
        noises0_output,
        noises1_output,
        disparities_origin,
        disparities_output,
        ground_truths)

    for idx, output in enumerate(outputs):
        image0_output, \
            image1_output, \
            noise0_output, \
            noise1_output, \
            disparity_origin, \
            disparity_output, \
            ground_truth  = output

        image_filename = '{:05d}.png'.format(idx)
        numpy_filename = '{:05d}.npy'.format(idx)

        # Save to disk
        Image.fromarray(np.uint8(image0_output * 255.0)).save(os.path.join(image0_output_path, image_filename))
        Image.fromarray(np.uint8(image1_output * 255.0)).save(os.path.join(image1_output_path, image_filename))

        np.save(os.path.join(noise0_output_path, numpy_filename), noise0_output)
        np.save(os.path.join(noise1_output_path, numpy_filename), noise1_output)

        np.save(os.path.join(disparity_origin_path, numpy_filename), disparity_origin)
        np.save(os.path.join(disparity_output_path, numpy_filename), disparity_output)

        np.save(os.path.join(ground_truth_path, numpy_filename), ground_truth)
