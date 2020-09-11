import cv2
from pytube import YouTube
from pathlib import Path


def download_video_youtube(url, save_dir, name='mv'):
    """
    Bruno Mars - That's What I Like
    https://www.youtube.com/watch?v=PMivT7MJ41M
    """
    yt = YouTube(url)
    yt.streams.first().download(save_dir, name)
    print(f'Downloaded video at {url} and saved to {save_dir}/{name}.mp4.')

def video_to_frame(video_dir, name='mv.mp4'):
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

def trucate_video(target, start=25, end=600):
    
    import os
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    os.mv(target, target + '_original')
    ffmpeg_extract_subclip(target + '_original', start, end, targetname=target)

def make_video_frame(url=None, video_dir='../data/source', truncate=False, record=False):
    ''' Download or record a video, truncate and transform to frames '''

    video_dir = Path(video_dir)
    video_dir.mkdir(exist_ok=True)

    if record:
        record_video(video_dir)
    else:
        download_video_youtube(url, video_dir)

    if truncate:
        trucate_video(target, truncate[0], truncate[1])
        
    video_to_frame(name, video_dir=video_dir)

if __name__ == "__main__":
    make_video_frame('https://www.youtube.com/watch?v=PMivT7MJ41M', 
                     '../data/source', record=False)    # Source video
    make_video_frame('https://www.youtube.com/watch?v=kyKNPPQW3bM', 
                     '../data/target', record=False)    # Target video
    