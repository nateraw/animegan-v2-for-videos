import gc
import math
import tempfile
from io import BytesIO

import gradio as gr
import numpy as np
import torch
from encoded_video import EncodedVideo, write_video
from PIL import Image
from torchvision.transforms.functional import center_crop, to_tensor

print("🧠 Loading Model...")
model = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device="cuda",
    progress=True,
)

def face2paint(model: torch.nn.Module, img: Image.Image, size: int = 512, device: str = 'cuda'):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)

    with torch.no_grad():
        input = to_tensor(img).unsqueeze(0) * 2 - 1
        output = model(input.to(device)).cpu()[0]

        output = (output * 0.5 + 0.5).clip(0, 1) * 255.0

    return output


# This function is taken from pytorchvideo!
def uniform_temporal_subsample(x: torch.Tensor, num_samples: int, temporal_dim: int = -3) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.
    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.
    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)


# This function is taken from pytorchvideo!
def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
    Returns:
        An x-like Tensor with scaled spatial dims.
    """
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))

    return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=False)


def inference_step(vid, start_sec, duration, out_fps):

    clip = vid.get_clip(start_sec, start_sec + duration)
    video_arr = torch.from_numpy(clip['video']).permute(3, 0, 1, 2)
    audio_arr = np.expand_dims(clip['audio'], 0)
    audio_fps = None if not vid._has_audio else vid._container.streams.audio[0].sample_rate

    x = uniform_temporal_subsample(video_arr, duration * out_fps)
    x = center_crop(short_side_scale(x, 512), 512)
    x /= 255.0
    x = x.permute(1, 0, 2, 3)
    with torch.no_grad():
        output = model(x.to('cuda')).detach().cpu()
        output = (output * 0.5 + 0.5).clip(0, 1) * 255.0
        output_video = output.permute(0, 2, 3, 1).numpy()

    return output_video, audio_arr, out_fps, audio_fps


def predict_fn(filepath, start_sec, duration, out_fps):
    # out_fps=12
    vid = EncodedVideo.from_path(filepath)
    for i in range(duration):
        print(f"🖼️ Processing step {i + 1}/{duration}...")
        video, audio, fps, audio_fps = inference_step(vid=vid, start_sec=i + start_sec, duration=1, out_fps=out_fps)
        gc.collect()
        if i == 0:
            video_all = video
            audio_all = audio
        else:
            video_all = np.concatenate((video_all, video))
            audio_all = np.hstack((audio_all, audio))

    print(f"💾 Writing output video...")
    write_video('out.mp4', video_all, fps=fps, audio_array=audio_all, audio_fps=audio_fps, audio_codec='aac')

    print(f"✅ Done!")
    del video_all
    del audio_all

    return 'out.mp4'


article = """
<p style='text-align: center'>
    <a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>Github Repo Pytorch</a>
</p>
"""

gr.Interface(
    predict_fn,
    inputs=[
        gr.inputs.Video(),
        gr.inputs.Slider(minimum=0, maximum=300, step=1, default=0),
        gr.inputs.Slider(minimum=1, maximum=5, step=1, default=2),
        gr.inputs.Slider(minimum=6, maximum=24, step=6, default=12),
    ],
    outputs=gr.outputs.Video(),
    title='AnimeGANV2 On Videos',
    description="Applying AnimeGAN-V2 to frames from video clips",
    article=article,
    enable_queue=True,
    examples=[
        ['obama.webm', 23, 5, 12],
    ],
    allow_flagging=False,
).launch(debug=True)
