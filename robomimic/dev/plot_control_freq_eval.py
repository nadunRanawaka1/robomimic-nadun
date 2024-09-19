import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_all_obs/20240918173401/logs/control_freq_eval_2024-09-19.pkl')

success_rate = df['Success_Rate']

success_rate = success_rate.tolist()[0]
print()