import moviepy
from moviepy.editor import VideoFileClip
from torchvision.tv_tensors import Video

video_file_list = ["/home/nadun/speedup_project/media/8_26/delta_model_2x.mp4",
                   "/home/nadun/speedup_project/media/8_26/4x_video.mp4",
                   "/home/nadun/speedup_project/media/8_26/3x_two.mp4"]

for video in video_file_list:
    video_clip = VideoFileClip(video)
    fn = video.split(".")[0]
    out_fn = f"{fn}.gif"
    video_clip.write_gif(out_fn)