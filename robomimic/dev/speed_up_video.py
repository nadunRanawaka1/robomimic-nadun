from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx


orig = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/aggregated_actions_with_gripper_check_100.mp4"
fast = "/media/nadun/Data/phd_project/robomimic/videos/lift_sped_up/aggregated_actions_with_gripper_check_100_5x.mp4"
clip = VideoFileClip(orig)

clip = clip.set_fps(clip.fps*5)

final = clip.fx(vfx.speedx, 5)
final.write_videofile(fast)