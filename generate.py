import os

import click
from typing import Union

import torch
import torchvision.transforms.functional as TVF
import numpy as np

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
@main.command(name='panorama')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', type=int, help='Set the random seed to start generating images from', default=42, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation factor for truncation trick', default=1, show_default=True)
@click.option('--num-frames', type=click.IntRange(min=1), help='Number of frames to generate (to be joined)', default=16, show_default=True)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Extra description to add to the output directory', default='', show_default=True)
def panorama(
        ctx: click.Context,
        network_pkl: str,
        seed: int,
        truncation_psi: float,
        num_frames: int,
        outdir: Union[str, os.PathLike],
        description: str
):
    """
    Example:
        python generate.py panorama --network=https://kaust-cair.s3.amazonaws.com/alis/lhq1024-snapshot.pkl --seed=0 \\
            --num-frames=10 --trunc=0.7

    """
    # Set the seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        G.eval()
        G.progressive_growing_update(100000)

    # Setup the run dir with the given description (if any)
    description = 'panorama' if len(description) == 0 else description
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
    max_shift = (num_frames_per_w * 2 - 1) * G.synthesis_cfg.patchwise.grid_size
    zs = torch.randn(num_ws, G.z_dim).to(device)
    mode_idx = 0
    modes_idx = (torch.ones(1, device=zs.device).repeat(num_ws).float() * mode_idx).long()
    ws = G.mapping(zs, c=None, modes_idx=modes_idx)

    # and truncation trick
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
    whole_img.save(os.path.join(run_dir, f'panorama_seed{seed}.png'))


np.random.seed(42)
torch.manual_seed(42)
torch.set_grad_enabled(False)

network_pkl = 'https://kaust-cair.s3.amazonaws.com/alis/lhq1024-snapshot.pkl'
device = 'cuda'

with dnnlib.util.open_url(network_pkl) as f:
    G = load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.eval()
    G.progressive_growing_update(100000)

for res in G.synthesis.block_resolutions:
    block = getattr(G.synthesis, f'b{res}')
    if hasattr(block, 'conv0'):
        block.conv0.use_noise = False
    block.conv1.use_noise = False


num_frames = 4
num_frames_per_w = G.synthesis_cfg.patchwise.w_coord_dist // 2
num_ws = num_frames // num_frames_per_w + 1
shifts = torch.arange(num_frames) * G.synthesis_cfg.patchwise.grid_size
w_range = 2 * num_frames_per_w * G.synthesis_cfg.patchwise.grid_size
max_shift = (num_frames_per_w * 2 - 1) * G.synthesis_cfg.patchwise.grid_size
zs = torch.randn(num_ws, G.z_dim).to(device)
mode_idx = 0
modes_idx = (torch.ones(1, device=zs.device).repeat(num_ws).float() * mode_idx).long()
ws = G.mapping(zs, c=None, modes_idx=modes_idx)

z_mean = torch.randn(1000, G.z_dim).to(device)
ws_proto = G.mapping(z_mean, c=None, modes_idx=modes_idx[0]).mean(dim=0, keepdim=True)

# Truncating
truncation_factor = 1.0
ws = ws * truncation_factor + (1 - truncation_factor) * ws_proto

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

whole_img = torch.cat(imgs, dim=2)
whole_img = TVF.to_pil_image(whole_img)

whole_img.show()
print()




# if __name__ == '__main__':
#     main()





# import os
# from PIL import Image
# import torch
# import torchvision.transforms.functional as TVF
#
# import numpy as np
# import scipy.misc
# import cv2
# from tqdm import tqdm
#
# # Load data
# frame_size = 512 # Determines the height (in pixels) of the video
# frames = [TVF.to_pil_image(img) for img in imgs]
# frames = [TVF.resize(img, frame_size, interpolation=Image.LANCZOS) for img in frames]
#
# step_size = 2 # Controls the camera movement speed (i.e. how many pixels we move from frame to frame)
# aspect_ratio = 4 # Aspect ratio of the video
# h = frame_size
# w = frame_size * aspect_ratio
# whole_img = torch.cat([TVF.to_tensor(im) for im in frames], dim=2)
# num_frames = (whole_img.shape[2] - frame_size * aspect_ratio) // step_size
# curr_offset = 0
#
# save_path = 'video.mp4'
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# video = cv2.VideoWriter(save_path, fourcc, 60, (w, h))
# for frame_idx in tqdm(range(num_frames)):
#     curr_offset += step_size
#     frame = whole_img[:, :, curr_offset:curr_offset + frame_size * aspect_ratio]
#     frame = TVF.to_pil_image(frame)
#     video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
#
# # Uncomment this line to release the memory.
# # It didn't work for me on centos and complained about installing additional libraries (which requires root access)
# # cv2.destroyAllWindows()
# video.release()