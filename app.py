from PIL import Image
import torch
import gradio as gr
import numpy as np
from encoded_video import EncodedVideo, write_video
from io import BytesIO

model2 = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device="cuda",
    progress=True,
    force_reload=True,
)
face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512, device="cuda",side_by_side=False
)

def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
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


def inference_video(video_file):
    out_fps = 12
    start_sec = 0
    duration = 2
    vid = EncodedVideo.from_path(video_file)
    clip = vid.get_clip(start_sec, start_sec + duration)
    video_arr = clip['video']
    audio_arr = np.expand_dims(clip['audio'], 0)
    audio_fps = None if not vid._has_audio else vid._container.streams.audio[0].sample_rate

    frames = uniform_temporal_subsample(torch.from_numpy(video_arr), duration * out_fps, 0).to(torch.uint8).numpy()

    out_frames = []
    for frame in frames:
        im = Image.fromarray(frame)
        out = face2paint(model2, im)
        out_frames.append(np.array(out))


    out_frames = np.array(out_frames)

    bytes_mp4 = bytes()
    out_file = BytesIO(bytes_mp4)

    # Add dummy file name to stream, as write_video will be looking for it
    out_file.name = "out.mp4"

    write_video(
        'out.mp4',
        out_frames,
        fps=out_fps,
        audio_array=audio_arr,
        audio_fps=audio_fps,
        audio_codec='aac'
    )
    return 'out.mp4'

gr.Interface(
    inference_video,
    inputs=gr.inputs.Video(),
    outputs=gr.outputs.Video(),
    title='AnimeGANV2 On Videos',
    description="Applying AnimeGAN-V2 to frame from video clips",
    article = "<p style='text-align: center'><a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>Github Repo Pytorch</a></p><p style='text-align: center'>samples from repo: <img src='https://user-images.githubusercontent.com/26464535/129888683-98bb6283-7bb8-4d1a-a04a-e795f5858dcf.gif' alt='animation'/> <img src='https://user-images.githubusercontent.com/26464535/137619176-59620b59-4e20-4d98-9559-a424f86b7f24.jpg' alt='animation'/><img src='https://user-images.githubusercontent.com/26464535/127134790-93595da2-4f8b-4aca-a9d7-98699c5e6914.jpg' alt='animation'/></p>",
    enable_queue=True,
    # examples=examples,
    allow_flagging=False
).launch(debug=True)