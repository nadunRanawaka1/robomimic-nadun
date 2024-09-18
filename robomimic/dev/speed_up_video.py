from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx


orig = "/home/nadun/Downloads/large_spat_shoulderview_right.mp4"
fast = "/home/nadun/Downloads/can_shoulderview_right_5x.mp4"
clip = VideoFileClip(orig)

clip = clip.set_fps(clip.fps*5)

final = clip.fx(vfx.speedx, 5)
final.write_videofile(fast)