import cv2
from pytube import YouTube
from pathlib import Path

def download_video_from_ytb(url, name, save_dir):
    """
    Bruno Mars - That's What I Like
    https://www.youtube.com/watch?v=PMivT7MJ41M
    """
    yt = YouTube(url)
    yt.streams.first().download(save_dir, 'mv')
    print(f'Downloaded video at {url} and saved to {save_dir}/{name}.')

def video_to_frame(name, video_dir):
    video_dir = Path(video_dir)
    video_dir.mkdir(exist_ok=True)
    frame_dir = video_dir.joinpath('images')
    frame_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_dir.joinpath(name).as_posix())
    i = 0
    while(cap.isOpened()):
        flag, frame = cap.read()
        if flag == False or i == 1000:
            break
        cv2.imwrite(frame_dir.joinpath(f'img_{i:04d}.png').as_posix(), frame)
        i += 1
    print(f'Converted video {video_dir}/{name} to frames at {frame_dir}.')

def record_video_to_frame(name, video_dir):
    video_dir = Path(video_dir)
    frame_dir = video_dir.joinpath('images')
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        is_capturing, _ = vc.read()
    else:
        is_capturing = False

    cnt = 0
    while is_capturing:
        is_capturing, img = vc.read()
        # img = img[:, 80:80+480]
        cv2.imwrite(frame_dir.joinpath(f'img_{cnt:04d}.png', img))
        cv2.imshow('video', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
        cnt += 1
    vc.release()
    cv2.destroyAllWindows()



import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage import io

import matplotlib.animation as ani
from IPython.display import HTML
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1000
%matplotlib inline


def _animate(nframe):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    source_img = io.imread(source_img_paths[nframe])
    ax1.imshow(source_img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    target_label = io.imread(target_label_paths[nframe])
    ax2.imshow(target_label)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    target_synth = io.imread(target_synth_paths[nframe])
    ax3.imshow(target_synth)
    ax3.set_xticks([])
    ax3.set_yticks([])    

def make_gif(source_dir='../data/source/test_img', 
             target_dir='../results/target/test_latest/images'):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    source_img_paths = sorted(source_dir.iterdir())
    target_synth_paths = sorted(target_dir.glob('*synthesized*'))
    target_label_paths = sorted(target_dir.glob('*input*'))

    assert len(source_img_paths) == len(target_synth_paths)
    assert len(target_synth_paths) == len(target_label_paths)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    anim = ani.FuncAnimation(fig, animate, frames=len(target_label_paths), interval=1000/24)
    plt.close()
    js_anim = HTML(anim.to_jshtml())
    anim.save("output.gif", writer="imagemagick")
