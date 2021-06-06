import os

import click
from typing import Union
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TVF
import numpy as np

from PIL import Image
import cv2

# We don't use this, but if we don't add it, Python raises ImportError, so ¯\_(ツ)_/¯
from distutils.dir_util import copy_tree
from training.networks import SynthesisLayer
from training.networks import PatchWiseSynthesisLayer

import dnnlib
from torch_utils.gen_utils import compress_video, make_run_dir
from scripts.legacy import load_network_pkl


# ----------------------------------------------------------------------------


# We group the different types of generation (panorama, video, other wacky stuff) into a main function
@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


# TODO: save individual frames?
# TODO: add resize option for final image
# TODO: come back to the first image (loop)
# TODO: let users decide either the video-step-size or video-length (in seconds)
@main.command(name='panorama')  # For now its own function apart from main, as more will be added below
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', type=int, help='Set the random seed to start generating images from', default=42, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation factor for truncation trick', default=0.7, show_default=True)  # not really truncation psi
@click.option('--num-frames', type=click.IntRange(min=4), help='Number of frames to generate that will be joined; must be a multiple of 4', default=16, show_default=True)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Extra description to add to the output directory', default=None, show_default=True)
# Video options
@click.option('--save-video', is_flag=True, help='Add flag to save a video')
@click.option('--video-height', 'frame_size', type=click.IntRange(min=1), help='Output height of the video (will use the original height by default)', default=None)
@click.option('--aspect-ratio', type=click.IntRange(min=1), help='Width to height ratio of the video (vertical is 1, this onlycontrols horizontal)', default=4, show_default=True)
@click.option('--video-step-size', 'step_size', type=click.IntRange(min=1), help='Camera movement speed: how many pixels to move from frame to frame', default=2, show_default=True)
@click.option('--video-fps', 'fps', type=click.IntRange(min=1), help='Video FPS (the lower, the slower to traverse the image and viceversa)', default=40, show_default=True)
@click.option('--compress-video', 'compress', help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)', is_flag=True)
def panorama(
        ctx: click.Context,
        network_pkl: str,
        seed: int,
        truncation_psi: float,
        num_frames: int,
        outdir: Union[str, os.PathLike],
        description: str,
        save_video: bool,
        frame_size: int,
        step_size: int,
        aspect_ratio: int,
        fps: int,
        compress: bool,
):
    """
    Example:
        python generate.py panorama --network=https://kaust-cair.s3.amazonaws.com/alis/lhq1024-snapshot.pkl --seed=0 \\
            --num-frames=20 --trunc=0.7
    """
    # Set the seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    # Sanity check: num_frames must be a multiple of 4
    if num_frames % 4 != 0:
        ctx.fail('Sorry, "--num-frames" must be a multiple of 4!')

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        G.eval()
        G.progressive_growing_update(100000)

    # Setup the run dir with the given description (if any)
    description = 'panorama' if description is None else description
    description = f'{description}-seed_{seed}-{num_frames}_frames-{truncation_psi}_truncation'
    run_dir = make_run_dir(outdir, description)

    # Turn off the noise
    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f'b{res}')
        if hasattr(block, 'conv0'):
            block.conv0.use_noise = False
        block.conv1.use_noise = False

    num_frames_per_w = G.synthesis_cfg.patchwise.w_coord_dist // 2
    num_ws = num_frames // num_frames_per_w + 1
    shifts = torch.arange(num_frames) * G.synthesis_cfg.patchwise.grid_size
    w_range = 2 * num_frames_per_w * G.synthesis_cfg.patchwise.grid_size
    max_shift = (num_frames_per_w * 2 - 1) * G.synthesis_cfg.patchwise.grid_size  # Not used???
    zs = torch.randn(num_ws, G.z_dim).to(device)
    mode_idx = 0
    modes_idx = (torch.ones(1, device=zs.device).repeat(num_ws).float() * mode_idx).long()
    ws = G.mapping(zs, c=None, modes_idx=modes_idx)

    # Generate z and w and do the truncation trick
    z_mean = torch.randn(1000, G.z_dim).to(device)
    ws_proto = G.mapping(z_mean, c=None, modes_idx=modes_idx[0]).mean(dim=0, keepdim=True)
    ws = ws * truncation_psi + (1 - truncation_psi) * ws_proto

    imgs = []
    curr_w_idx = 1
    curr_ws = ws[curr_w_idx].unsqueeze(0)
    curr_ws_context = torch.stack([ws[curr_w_idx - 1].unsqueeze(0), ws[curr_w_idx + 1].unsqueeze(0)], dim=1)

    for shift in shifts:
        if shift > 0 and shift % w_range == 0:
            curr_w_idx += 2
            curr_ws = ws[curr_w_idx].unsqueeze(0)
            curr_ws_context = torch.stack([ws[curr_w_idx - 1].unsqueeze(0), ws[curr_w_idx + 1].unsqueeze(0)], dim=1)

        curr_left_borders_idx = torch.zeros(1, device=zs.device).long() + (shift % w_range)
        img = G.synthesis(curr_ws, ws_context=curr_ws_context, left_borders_idx=curr_left_borders_idx, noise='const')
        imgs.append(img[0].cpu().clamp(-1, 1) * 0.5 + 0.5)

    # Form the whole panorama image by appending the generated images
    whole_img = TVF.to_pil_image(torch.cat(imgs, dim=2))
    whole_img.save(os.path.join(run_dir, f'panorama_seed-{seed}.png'))

    if save_video:
        print('Saving the video...')
        # Load data
        frame_size = whole_img.size[1] if frame_size is None else frame_size  # The height (in pixels) of the video
        frames = [TVF.to_pil_image(img) for img in imgs]
        frames = [TVF.resize(img, frame_size, interpolation=Image.LANCZOS) for img in frames]

        h = frame_size
        w = frame_size * aspect_ratio
        whole_img = torch.cat([TVF.to_tensor(im) for im in frames], dim=2)
        num_frames = (whole_img.shape[2] - frame_size * aspect_ratio) // step_size
        curr_offset = 0

        # Set the video name
        video_name = f'panorama_video-{aspect_ratio}x1'
        video_name = f'{video_name}-{fps}fps'

        save_path = os.path.join(run_dir, f'{video_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(filename=save_path, fourcc=fourcc, fps=fps, frameSize=(w, h))
        for frame_idx in tqdm(range(num_frames)):
            curr_offset += step_size
            frame = whole_img[:, :, curr_offset:curr_offset + frame_size * aspect_ratio]
            frame = TVF.to_pil_image(frame)
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # Uncomment this line to release the memory.
        # It didn't work for me on centos and complained about installing additional libraries (which requires root access)
        # cv2.destroyAllWindows()
        video.release()

        # Compress the video with ffmpeg-python (must be installed via pip, is not always perfect)
        if compress:
            compress_video(original_video=save_path, original_video_name=video_name, outdir=run_dir, ctx=ctx)


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
