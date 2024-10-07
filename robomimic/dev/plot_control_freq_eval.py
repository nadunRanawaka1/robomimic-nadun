import pandas as pd
import matplotlib.pyplot as plt
import pickle

from docutils.nodes import label
from matplotlib.pyplot import ylabel
import numpy as np


def plot_replay(data_fp=None):
    data_fp = '/media/nadun/Data/phd_project/experiment_logs/control_freq_eval/can_eval.pkl'

    with open(data_fp, 'rb') as f:

        data = pickle.load(f)

    print()

    X = []
    pos_y_1 = []
    pos_y_2 = []
    rot_y_1 = []
    rot_y_2 = []
    for freq in data:
        X.append(freq)
        errors = data[freq]
        pos_errors = errors['pos_errors']
        rot_errors = errors['rot_errors']

        mean_pos_error = sum(pos_errors)/len(pos_errors)
        mean_rot_error = sum(rot_errors)/len(rot_errors)

        max_pos_errors = errors['max_pos_errors']
        max_rot_errors = errors['max_rot_errors']

        max_pos_error = max(max_pos_errors)
        max_rot_error = max(max_rot_errors)

        pos_y_1.append(mean_pos_error)
        pos_y_2.append(max_pos_error)

        rot_y_1.append(mean_rot_error)
        rot_y_2.append(max_rot_error)

    fig, axs = plt.subplots(2)
    axs[0].plot(X, pos_y_1, label='mean')
    axs[0].plot(X, pos_y_2, ls='None', color='red', markersize=8, marker='X', label='max')
    axs[0].set_title("Position Error")
    axs[0].set(ylabel='Error (m)')

    axs[1].plot(X, rot_y_1, label='mean')
    axs[1].plot(X, rot_y_2, ls='None', color='red', markersize=8, marker='X', label='max')
    axs[1].set_title("Rotation Error")
    axs[1].set(ylabel="Error (Quaternion magnitude)")

    fig.supxlabel('Control Freq')

    plt.legend()
    plt.show()

def plot_model(data_fp):

    df = pd.read_pickle(data_fp)

    control_freqs = [10,20,30,40,50,60,70,80,90]
    df['control_freq'] = control_freqs

    X = []
    success_rates = []
    sim_steps = []

    for index, row in df.iterrows():
        cf = row['control_freq']
        X.append(cf)

        succ_rate = np.array(row['Success_Rate'])
        succ = np.mean(succ_rate)
        success_rates.append(succ)
        horizon = np.array(row['Horizon'])

        mask = succ_rate == 1.0
        horizon = horizon[mask]
        sim_steps.append(np.mean(horizon) * (1000/cf))

    fig, axs = plt.subplots(2)
    axs[0].plot(X, success_rates, label="Success Rate")
    axs[1].plot(X, sim_steps, color='red', label="Sim steps per success")

    fig.suptitle("Eval of OSC models over different control freqs")
    fig.supxlabel('Control Freq')
    plt.legend()

    plt.show()

    print()


plot_model('/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_all_obs/20240918173401/logs/control_freq_eval_2024-09-19.pkl')