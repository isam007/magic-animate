import argparse
import imageio
import os, datetime
import numpy as np
import gradio as gr
from PIL import Image
from subprocess import PIPE, run

base_dir = os.path.dirname(os.path.abspath(__file__))
demo_dir = os.path.join(base_dir, "demo")
tmp_dir = os.path.join(demo_dir, "tmp")
outputs_dir = os.path.join(demo_dir, "outputs")

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

def animate(reference_image, motion_sequence, seed, steps, guidance_scale):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    animation_path = os.path.join(outputs_dir, f"{time_str}.mp4")
    save_path = os.path.join(tmp_dir, "input_reference_image.png")
    Image.fromarray(reference_image).save(save_path)
    command = f"python -m demo.animate_dist --reference_image {save_path} --motion_sequence {motion_sequence} --random_seed {seed} --step {steps} --guidance_scale {guidance_scale} --save_path {animation_path}"
    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return animation_path

with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/magic-research/magic-animate" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model</h1>
            <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href="https://arxiv.org/abs/2311.16498"><img src="https://img.shields.io/badge/Arxiv-2311.16498-red"></a>
                <a href='https://showlab.github.io/magicanimate'><img src='https://img.shields.io/badge/Project_Page-MagicAnimate-green' alt='Project Page'></a>
                <a href='https://github.com/magic-research/magic-animate'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
            </div>
        </div>
        </div>
        """
    )
    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)
    
    with gr.Row():
        reference_image  = gr.Image(label="Reference Image")
        motion_sequence  = gr.Video(format="mp4", label="Motion Sequence")
        
        with gr.Column():
            random_seed         = gr.Textbox(label="Random seed", value=1, info="default: -1")
            sampling_steps      = gr.Textbox(label="Sampling steps", value=25, info="default: 25")
            guidance_scale      = gr.Textbox(label="Guidance scale", value=7.5, info="default: 7.5")
            submit              = gr.Button("Animate")

    def read_video(video, size=512):
        size = int(size)
        reader = imageio.get_reader(video)
        frames = []
        for img in reader:
            frames.append(np.array(Image.fromarray(img).resize((size, size))))
        
        save_path = os.path.join(tmp_dir, "input_motion_sequence.mp4")
        imageio.mimwrite(save_path, frames, fps=25)
        return save_path
    
    def read_image(image, size=512):
        img = np.array(Image.fromarray(image).resize((size, size)))
        return img
        
    # when user uploads a new video
    motion_sequence.upload(
        read_video,
        motion_sequence,
        motion_sequence
    )
    # when `first_frame` is updated
    reference_image.upload(
        read_image,
        reference_image,
        reference_image
    )
    # when the `submit` button is clicked
    submit.click(
        animate,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
        animation
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ["inputs/applications/source_image/monalisa.png", "inputs/applications/driving/densepose/running.mp4"], 
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )

demo.launch(share=False,inbrowser=True)
