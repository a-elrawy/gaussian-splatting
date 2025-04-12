#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, score_camera_uncertainty
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import random

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def select_next_view(scene, gaussians, pipeline, background, available_view_indices, available_view_stack, iteration, uncertainty_update_from_iter, num_candidates=10, selection_strategy="random"):
    """
    Selects the next camera view based on uncertainty.

    Args:
        scene: The scene object.
        gaussians: The GaussianModel.
        pipeline: Rendering pipeline parameters.
        background: Background color tensor.
        available_view_indices: List of indices of views not yet used.
        available_view_stack: List of Camera objects not yet used.
        iteration: Current training iteration.
        uncertainty_update_from_iter: Iteration from which uncertainty updates start (used for random selection threshold).
        num_candidates: How many candidate views to evaluate.
        selection_strategy: Method to choose the best view ('highest_uncertainty', 'random').

    Returns:
        Tuple: (selected_camera, selected_index_in_original_list, updated_available_indices, updated_available_stack)
               Returns (None, -1, ...) if no views are available.
    """
    if not available_view_indices:
        return None, -1, available_view_indices, available_view_stack

    # Always use random selection for the first few iterations for stability
    if selection_strategy == "random" or iteration < uncertainty_update_from_iter: # Use passed threshold
        rand_idx_in_available = randint(0, len(available_view_indices) - 1)
        original_idx = available_view_indices.pop(rand_idx_in_available)
        selected_cam = available_view_stack.pop(rand_idx_in_available)
        print(f"[Iter {iteration}] Selected view {original_idx} randomly (Iter < {uncertainty_update_from_iter} or strategy='random').")
        return selected_cam, original_idx, available_view_indices, available_view_stack

    elif selection_strategy == "highest_uncertainty":
        # 1. Generate Candidate Views
        num_available = len(available_view_indices)
        actual_num_candidates = min(num_candidates, num_available)
        if actual_num_candidates <= 0:
            return None, -1, available_view_indices, available_view_stack

        candidate_indices_in_available = random.sample(range(num_available), actual_num_candidates)
        candidate_cams = [available_view_stack[i] for i in candidate_indices_in_available]
        candidate_original_indices = [available_view_indices[i] for i in candidate_indices_in_available]

        candidate_scores = []

        # 2. Render Uncertainty and Score Candidates
        print(f"[Iter {iteration}] Evaluating {len(candidate_cams)} candidates for uncertainty...")
        with torch.no_grad():
            for cam_idx, cam in enumerate(candidate_cams):
                try:
                    render_pkg = render(cam, gaussians, pipeline, background, render_uncertainty=True)
                    uncertainty_map = render_pkg.get("uncertainty_map")

                    if uncertainty_map is not None and uncertainty_map.numel() > 0:
                        score = score_camera_uncertainty(uncertainty_map, method="mean")
                        candidate_scores.append(score)
                    else:
                        print(f"Warning: Failed to render uncertainty or map empty for candidate view {cam.uid} (Orig Idx {candidate_original_indices[cam_idx]}). Assigning low score.")
                        candidate_scores.append(-1.0)
                except Exception as e:
                    print(f"Error rendering uncertainty for candidate view {cam.uid} (Orig Idx {candidate_original_indices[cam_idx]}): {e}. Assigning low score.")
                    candidate_scores.append(-1.0)

        if not candidate_scores:
            print(f"Warning: No candidate views could be scored at iteration {iteration}. Selecting randomly.")
            rand_idx_in_available = randint(0, len(available_view_indices) - 1)
            original_idx = available_view_indices.pop(rand_idx_in_available)
            selected_cam = available_view_stack.pop(rand_idx_in_available)
            return selected_cam, original_idx, available_view_indices, available_view_stack

        best_candidate_local_idx = np.argmax(candidate_scores)
        selected_cam_local_idx_in_available = candidate_indices_in_available[best_candidate_local_idx]

        selected_cam = available_view_stack.pop(selected_cam_local_idx_in_available)
        original_idx = available_view_indices.pop(selected_cam_local_idx_in_available)

        print(f"[Iter {iteration}] Selected view {original_idx} with uncertainty score: {candidate_scores[best_candidate_local_idx]:.4f} (Max score among candidates)")

        return selected_cam, original_idx, available_view_indices, available_view_stack

    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack_all = scene.getTrainCameras().copy()
    viewpoint_indices_all = list(range(len(viewpoint_stack_all)))

    available_view_stack = viewpoint_stack_all.copy()
    available_view_indices = viewpoint_indices_all.copy()

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_uncertainty_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    render_pkg_gui = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                    net_image = render_pkg_gui["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not available_view_indices:
            print(f"Replenishing view pool at iteration {iteration}")
            available_view_stack = viewpoint_stack_all.copy()
            available_view_indices = viewpoint_indices_all.copy()
            if not available_view_indices:
                 sys.exit("Error: No training cameras available to replenish pool.")

        viewpoint_cam, vind, available_view_indices, available_view_stack = select_next_view(
            scene, gaussians, pipe, background,
            available_view_indices, available_view_stack, iteration,
            uncertainty_update_from_iter=opt.uncertainty_update_from_iter,
            num_candidates=dataset.active_learning_candidates,
            selection_strategy=dataset.active_learning_strategy
        )

        if viewpoint_cam is None:
            print("Warning: No view selected. Skipping iteration.")
            continue

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_uncertainty=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if visibility_filter is not None and visibility_filter.any():
            num_gaussians = gaussians.get_xyz.shape[0]
            current_device = gaussians.get_xyz.device
            if gaussians._visibility_counter is None or gaussians._visibility_counter.shape[0] != num_gaussians:
                gaussians._visibility_counter = torch.zeros(num_gaussians, dtype=torch.long, device=current_device)
            else:
                gaussians._visibility_counter = gaussians._visibility_counter.to(current_device)

            visibility_filter_bool = visibility_filter.bool().squeeze().to(current_device)

            if visibility_filter_bool.ndim == 1 and visibility_filter_bool.shape[0] == num_gaussians:
                gaussians._visibility_counter[visibility_filter_bool] += 1
            else:
                 print(f"Warning: visibility_filter shape mismatch during counter update. "
                       f"Filter shape: {visibility_filter_bool.shape}, Expected: ({num_gaussians},). Skipping update.")

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            if alpha_mask.shape[1:] == image.shape[1:]:
                image *= alpha_mask
            else:
                print(f"Warning: Alpha mask shape {alpha_mask.shape} incompatible with image shape {image.shape}. Skipping masking.")

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            image_ssim = torch.clamp(image, 0.0, 1.0).unsqueeze(0)
            gt_image_ssim = torch.clamp(gt_image, 0.0, 1.0).unsqueeze(0)
            ssim_value = fused_ssim(image_ssim, gt_image_ssim)
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        Ll1depth_pure = 0.0
        current_depth_weight = depth_l1_weight(iteration)
        if current_depth_weight > 0 and viewpoint_cam.depth_reliable and viewpoint_cam.invdepthmap is not None and viewpoint_cam.depth_mask is not None:
            try:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                if invDepth.shape == mono_invdepth.shape and invDepth.shape == depth_mask.shape:
                    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                    Ll1depth_loss_term = current_depth_weight * Ll1depth_pure
                    loss += Ll1depth_loss_term
                    Ll1depth = Ll1depth_loss_term.item()
                else:
                    print(f"Warning: Depth/mask shape mismatch. Depth: {invDepth.shape}, MonoInvDepth: {mono_invdepth.shape}, Mask: {depth_mask.shape}. Skipping depth loss.")
                    Ll1depth = 0.0
            except KeyError:
                print("Warning: 'depth' key not found in render_pkg. Skipping depth loss.")
                Ll1depth = 0.0
            except Exception as e:
                print(f"Error during depth loss calculation: {e}. Skipping depth loss.")
                Ll1depth = 0.0
        else:
            Ll1depth = 0.0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            current_avg_uncertainty = gaussians.get_uncertainty.mean().item() if gaussians.get_uncertainty.numel() > 0 else 0.0
            ema_uncertainty_for_log = 0.4 * current_avg_uncertainty + 0.6 * ema_uncertainty_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Depth": f"{ema_Ll1depth_for_log:.{5}f}",
                    "Uncert": f"{ema_uncertainty_for_log:.{3}f}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, current_avg_uncertainty)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                num_gaussians_current = gaussians.get_xyz.shape[0]
                valid_filter = visibility_filter is not None and visibility_filter.ndim == 1 and visibility_filter.shape[0] == num_gaussians_current
                valid_radii = radii is not None and radii.ndim == 1 and radii.shape[0] == num_gaussians_current

                if valid_filter and valid_radii:
                    visibility_filter_bool = visibility_filter.bool().to(gaussians.max_radii2D.device)
                    radii_dev = radii.to(gaussians.max_radii2D.device)
                    gaussians.max_radii2D[visibility_filter_bool] = torch.max(gaussians.max_radii2D[visibility_filter_bool], radii_dev[visibility_filter_bool])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter_bool)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    current_radii_for_prune = radii if valid_radii else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, current_radii_for_prune)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    if iteration > opt.densify_from_iter:
                         gaussians.reset_opacity()

            if iteration % opt.uncertainty_update_interval == 0 and iteration >= opt.uncertainty_update_from_iter:
                lambda_err = opt.lambda_err
                gamma_view = opt.gamma_view
                min_views = opt.min_views_for_uncertainty
                gaussians.update_uncertainty(lambda_err, gamma_view, min_views_for_update=min_views, iteration=iteration)

            if iteration < opt.iterations:
                num_gaussians_opt = gaussians.get_xyz.shape[0]
                valid_radii_opt = radii is not None and radii.ndim == 1 and radii.shape[0] == num_gaussians_opt

                if use_sparse_adam:
                    if valid_radii_opt:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, num_gaussians_opt)
                    else:
                        print(f"Warning: Radii invalid for sparse adam step at iteration {iteration}. Stepping all parameters.")
                        gaussians.optimizer.step()
                else:
                    gaussians.optimizer.step()

                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss_func, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, current_avg_uncertainty):
    if tb_writer:
        tb_writer.add_scalar('Loss/train_l1', Ll1.item(), iteration)
        tb_writer.add_scalar('Loss/train_total', loss.item(), iteration)
        tb_writer.add_scalar('Timing/iter_time_ms', elapsed, iteration)
        tb_writer.add_scalar('Metrics/avg_uncertainty', current_avg_uncertainty, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train_subset', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    try:
                        render_pkg_val = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        image = torch.clamp(render_pkg_val["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                        if tb_writer and (idx < 5):
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.uid}/render", image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(f"{config['name']}_view_{viewpoint.uid}/ground_truth", gt_image[None], global_step=iteration)

                        l1_test += l1_loss_func(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    except Exception as e:
                        print(f"Error during validation rendering for view {viewpoint.uid}: {e}")
                        continue

                if len(config['cameras']) > 0:
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if tb_writer:
                        tb_writer.add_scalar(f"Val_{config['name']}/l1", l1_test, iteration)
                        tb_writer.add_scalar(f"Val_{config['name']}/psnr", psnr_test, iteration)
                else:
                    print(f"\n[ITER {iteration}] Evaluating {config['name']}: No valid views rendered.")

        if tb_writer:
            tb_writer.add_histogram("Scene/opacity", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('Scene/points', scene.gaussians.get_xyz.shape[0], iteration)
            if scene.gaussians.get_uncertainty.numel() > 0:
                tb_writer.add_histogram("Scene/uncertainty", scene.gaussians.get_uncertainty, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--uncertainty_update_interval", type=int, default=100)
    parser.add_argument("--uncertainty_update_from_iter", type=int, default=1000)
    parser.add_argument("--lambda_err", type=float, default=0.0)
    parser.add_argument("--gamma_view", type=float, default=0.1)
    parser.add_argument("--min_views_for_uncertainty", type=int, default=1)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_params = lp.extract(args)
    opt_params = op.extract(args)
    pipe_params = pp.extract(args)

    opt_params.uncertainty_update_interval = args.uncertainty_update_interval
    opt_params.uncertainty_update_from_iter = args.uncertainty_update_from_iter
    opt_params.lambda_err = args.lambda_err
    opt_params.gamma_view = args.gamma_view
    opt_params.min_views_for_uncertainty = args.min_views_for_uncertainty

    print("Optimizing " + model_params.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(model_params, opt_params, pipe_params, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
