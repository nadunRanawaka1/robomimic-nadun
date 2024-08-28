from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx


orig = "/home/nadun/Downloads/all_views_as_agentview.MOV"
fast = "/home/nadun/Downloads/all_views_as_agentview_5x.mp4"
clip = VideoFileClip(orig)

clip = clip.set_fps(clip.fps*5)

final = clip.fx(vfx.speedx, 5)
final.write_videofile(fast)