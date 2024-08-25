import h5py
from copy import deepcopy
import nexusformat.nexus as nx

demo_fn = "/nethome/nkra3/flash7/ATRP_AC_Learning/datasets/speed_up_demos/demo_put_strawberry_in_bowl.hdf5"
demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

demo_f = h5py.File(demo_fn, 'r')
demo_file = demo_f['data']

# counter = 0
# for demo in demo_file:
#     print(f"processing demo: {demo}")
#     demo = demo_file[demo]
#     delta_joint_actions = deepcopy(demo['joint_position_actions'][:])

#     joint_obs = demo['obs/joint_positions'][:]
#     delta_joint_actions[:, :-1] = delta_joint_actions[:, :-1] - joint_obs
#     demo.create_dataset("delta_joint_actions", data = delta_joint_actions)



delta_joint_actions = demo_file['demo_10/delta_joint_actions'][:]
print(delta_joint_actions)