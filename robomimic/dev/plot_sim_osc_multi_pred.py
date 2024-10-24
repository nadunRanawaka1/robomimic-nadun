import pickle
import matplotlib.pyplot as plt

### Some constants

PLOT_FEATURE = 2 # which part of the pred to plot

with open("/media/nadun/Data/phd_project/experiment_logs/sim_multi_eval/can_image_joint_position.pkl", 'rb') as f:
    data = pickle.load(f)

demo = data['demo_0']
preds = demo['preds']
execute_n_actions = demo["execute_n_actions"]

### Plot the predictions over time
x_lists = []
y_lists = []

for i, pred in enumerate(preds):
    X = []
    y = []
    start_timestep = i*execute_n_actions
    for j in range(pred.shape[0]):
        X.append(start_timestep+j)
        y.append(pred[j, PLOT_FEATURE])
    x_lists.append(X)
    y_lists.append(y)

for idx, X in enumerate(x_lists):
    plt.plot(X, y_lists[idx])


plt.ylabel("Predicted x position")
plt.xlabel("Timestep")
plt.show()

print()