import os
import re
from typing import Union
import click


# ----------------------------------------------------------------------------


def compress_video(
        original_video: Union[str, os.PathLike],
        original_video_name: Union[str, os.PathLike],
        outdir: Union[str, os.PathLike],
        ctx: click.Context) -> None:
    """ Helper function to compress the original_video using ffmpeg-python. moviepy creates huge videos, so use
        ffmpeg to 'compress' it (won't be perfect, 'compression' will depend on the video dimensions). ffmpeg
        can also be used to e.g. resize the video, make a GIF, save all frames in the video to the outdir, etc.
    """
    try:
        import ffmpeg
    except (ModuleNotFoundError, ImportError):
        ctx.fail('Missing ffmpeg! Install it via "pip install ffmpeg-python"')

    print('Compressing the video...')
    resized_video_name = os.path.join(outdir, f'{original_video_name}-compressed.mp4')
    ffmpeg.input(original_video).output(resized_video_name).run(capture_stdout=True, capture_stderr=True)
    print('Success!')


# ----------------------------------------------------------------------------


def make_run_dir(outdir: Union[str, os.PathLike], desc: str, dry_run: bool = False) -> str:
    """Reject modernity, return to automatically create the run dir."""
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):  # sanity check, but click.Path() should clear this one
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1  # start with 00000
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)  # make sure it doesn't already exist

    # Don't create the dir if it's a dry-run
    if not dry_run:
        print('Creating output directory...')
        os.makedirs(run_dir)
    return run_dir
