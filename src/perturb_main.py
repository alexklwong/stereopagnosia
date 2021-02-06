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
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import torch
import datasets, data_utils, eval_utils
import global_constants as settings
from log_utils import log
from stereo_model import StereoModel
from perturb_model import PerturbationsModel


def run(image0_path,
        image1_path,
        ground_truth_path=None,
        # Run settings
        n_height=settings.N_HEIGHT,
        n_width=settings.N_WIDTH,
        # Perturb method settings
        perturb_method=settings.PERTURB_METHOD,
        perturb_mode=settings.PERTURB_MODE,
        output_norm=settings.OUTPUT_NORM,
        n_step=settings.N_STEP,
        learning_rate=settings.LEARNING_RATE,
        momentum=settings.MOMENTUM,
        probability_diverse_input=settings.PROBABILITY_DIVERSE_INPUT,
        # Stereo model settings
        stereo_method=settings.STEREO_METHOD,
        stereo_model_restore_path=settings.STEREO_MODEL_RESTORE_PATH,
        # Output settings
        output_path='',
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

    n_sample = len(image0_paths)

    assert n_sample == len(image1_paths)

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
    log('n_height=%d  n_width=%d  output_norm=%.3f' %
        (n_height, n_width, output_norm),
        log_path)
    log('perturb_method=%s  perturb_mode=%s' %
        (perturb_method, perturb_mode),
        log_path)
    log('n_step=%s  learning_rate=%.1e' %
        (n_step, learning_rate),
        log_path)
    log('momentum=%.2f' %
        (momentum),
        log_path)
    log('probability_diverse_input=%.2f' %
        (probability_diverse_input),
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

    time_start = time.time()
    time_per_frame = 0.0

    disparities_origin = []
    noises0_output = []
    noises1_output = []
    images0_output = []
    images1_output = []
    disparities_output = []
    ground_truths = []

    for idx, (image0, image1, ground_truth) in enumerate(dataloader):

        if device.type == settings.CUDA:
            image0 = image0.cuda()
            image1 = image1.cuda()
            ground_truth = ground_truth.cuda()

        if len(image0.shape) == 4 and image0.shape[0] == 1:
            ground_truth = torch.unsqueeze(ground_truth, dim=0)

        # Initialize perturbations
        perturb_model = PerturbationsModel(
            perturb_method=perturb_method,
            perturb_mode=perturb_mode,
            output_norm=output_norm,
            n_step=n_step,
            learning_rate=learning_rate,
            momentum=momentum,
            probability_diverse_input=probability_diverse_input,
            device=device)

        # Forward through through stereo network
        disparity_origin = stereo_model.forward(image0, image1)

        # Save original disparity
        disparities_origin.append(
            np.squeeze(disparity_origin.detach().cpu().numpy()))

        ground_truths.append(
            np.squeeze(ground_truth.cpu().numpy()))

        time_per_frame_start = time.time()

        # Optimize perturbations for the stereo pair and model
        noise0_output, noise1_output, image0_output, image1_output = perturb_model.forward(
            stereo_model=stereo_model,
            image0=image0,
            image1=image1,
            ground_truth=ground_truth)

        time_per_frame = time_per_frame + (time.time() - time_per_frame_start)

        # Forward through network again
        disparity_output = stereo_model.forward(image0_output, image1_output)

        loss_func = torch.nn.L1Loss()
        loss = loss_func(disparity_output, ground_truth)

        # Save outputs
        noises0_output.append(
            np.transpose(np.squeeze(noise0_output.detach().cpu().numpy()), (1, 2, 0)))
        noises1_output.append(
            np.transpose(np.squeeze(noise1_output.detach().cpu().numpy()), (1, 2, 0)))
        images0_output.append(
            np.transpose(np.squeeze(image0_output.detach().cpu().numpy()), (1, 2, 0)))
        images1_output.append(
            np.transpose(np.squeeze(image1_output.detach().cpu().numpy()), (1, 2, 0)))
        disparities_output.append(
            np.squeeze(disparity_output.detach().cpu().numpy()))

        # Log results
        time_elapse = (time.time() - time_start) / 3600

        log('Sample={:3}/{:3}  L1 Loss={:.5f}  Time Elapsed={:.2f}h'.format(
            idx + 1, n_sample, loss.item(), time_elapse),
            log_path)

        # Clean up
        del perturb_model
        del image0, image1
        del image0_output, image1_output
        del noise0_output, noise1_output
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

    log('Time per frame: {:.2f}s'.format(time_per_frame / len(dataloader)), log_path)
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

def validate(noises0_output,
             noises1_output,
             disparities_output,
             ground_truths,
             device,
             log_path):

    n_sample = len(disparities_output)

    assert n_sample == len(noises0_output)
    assert n_sample == len(noises1_output)
    assert n_sample == len(ground_truths)

    # Noise metrics
    noise0_l0pix = np.zeros(n_sample)
    noise0_l1pix = np.zeros(n_sample)
    noise1_l0pix = np.zeros(n_sample)
    noise1_l1pix = np.zeros(n_sample)

    # Disparity metrics
    disparity_mae = np.zeros(n_sample)
    disparity_rmse = np.zeros(n_sample)
    disparity_d1 = np.zeros(n_sample)

    data = zip(noises0_output, noises1_output, disparities_output, ground_truths)

    for idx, (noise0_output, noise1_output, disparity_output, ground_truth) in enumerate(data):

        # Compute noise metrics
        noise0_l0pix[idx] = eval_utils.lp_norm(noise0_output, p=0)
        noise0_l1pix[idx] = eval_utils.lp_norm(noise0_output, p=1, axis=-1)
        noise1_l0pix[idx] = eval_utils.lp_norm(noise1_output, p=0)
        noise1_l1pix[idx] = eval_utils.lp_norm(noise1_output, p=1, axis=-1)

        # Mask out invalid ground truth
        mask = np.logical_and(ground_truth > 0.0, ~np.isnan(ground_truth))

        # Compute disparity metrics
        disparity_mae[idx] = \
            eval_utils.mean_abs_err(disparity_output[mask], ground_truth[mask])
        disparity_rmse[idx] = \
            eval_utils.root_mean_sq_err(disparity_output[mask], ground_truth[mask])
        disparity_d1[idx] = \
            eval_utils.d1_error(disparity_output[mask], ground_truth[mask])

    # Noise metrics
    noise0_l0pix = np.mean(noise0_l0pix)
    noise0_l1pix = np.mean(noise0_l1pix)

    noise1_l0pix = np.mean(noise1_l0pix)
    noise1_l1pix = np.mean(noise1_l1pix)

    # Disparity metrics
    disparity_mae_std = np.std(disparity_mae)
    disparity_mae_mean = np.mean(disparity_mae)

    disparity_rmse_std = np.std(disparity_rmse)
    disparity_rmse_mean = np.mean(disparity_rmse)

    disparity_d1_std = np.std(disparity_d1 * 100.0)
    disparity_d1_mean = np.mean(disparity_d1 * 100.0)

    log('Validation results:',
        log_path)

    log('{:<14}  {:>10}  {:>10}'.format(
        'Noise0:', 'L0 Pixel', 'L1 Pixel'),
        log_path)
    log('{:<14}  {:10.4f}  {:10.4f}'.format(
        '', noise0_l0pix, noise0_l1pix),
        log_path)

    log('{:<14}  {:>10}  {:>10}'.format(
        'Noise1:', 'L0 Pixel', 'L1 Pixel'),
        log_path)
    log('{:<14}  {:10.4f}  {:10.4f}'.format(
        '', noise1_l0pix, noise1_l1pix),
        log_path)

    log('{:<14}  {:>10}  {:>10}'.format(
        'Disparity:', 'MAE', '+/-'),
        log_path)
    log('{:<14}  {:>10.4f}  {:>10.4f}'.format(
        '', disparity_mae_mean, disparity_mae_std),
        log_path)

    log('{:<14}  {:>10}  {:>10}'.format(
        '', 'RMSE', '+/-', ),
        log_path)
    log('{:<14}  {:>10.4f}  {:>10.4f}'.format(
        '', disparity_rmse_mean, disparity_rmse_std),
        log_path)

    log('{:<14}  {:>10}  {:>10}'.format(
        '', 'D1-Error', '+/-', ),
        log_path)
    log('{:<14}  {:>10.4f}  {:>10.4f}'.format(
        '', disparity_d1_mean, disparity_d1_std),
        log_path)
