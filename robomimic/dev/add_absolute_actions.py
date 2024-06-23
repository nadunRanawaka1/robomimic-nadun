import nexusformat.nexus as nx
import numpy as np
import h5py
import robomimic.utils.transform_utils as T



def delta_axis_angle_to_absolute(delta_angles, quats):
    ret = []
    for i in range(quats.shape[0]):
        quat = quats[i]
        delta_aa = delta_angles[i]

        start_rot = T.quat2mat(quat)
        axis, angle = T.vec2axisangle(delta_aa)
        delta_rot = T.axisangle2mat(axis, angle)
        abs_rot_mat = np.dot(delta_rot, start_rot)  # Final absolute rot mat

        abs_axis, abs_angle = T.mat2axisangle(abs_rot_mat)
        abs_aa = T.axisangle2vec(abs_axis, abs_angle)
        if any(np.isnan(abs_aa)):
            print()
        ret.append(abs_aa)

    return np.array(ret)


demo_fn  = "/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/tool_hang/ph/all_obs_v141.hdf5"
# demo = nx.nxload(demo_fn)
# print(demo.tree)

demo_f = h5py.File(demo_fn, "a")
demo_file = demo_f['data']
env_args = demo_file.attrs["env_args"]
print(env_args)

for demo in demo_file:
    demo = demo_file[demo]
    delta_actions = demo['actions'][:]

    # Create absolute position action
    delta_pos = delta_actions[:, 0:3]
    pos = demo['obs/robot0_eef_pos'][:]
    target_pos = delta_pos + pos

    # Create absolute axis angle rotation action
    delta_aa = delta_actions[:, 3:6]
    quat = demo['obs/robot0_eef_quat'][:]
    target_axis_angles = delta_axis_angle_to_absolute(delta_aa, quat)

    # Create new absolute action array:
    gripper_act = delta_actions[:, -1, np.newaxis]
    absolute_actions = np.concatenate((target_pos, target_axis_angles, gripper_act), axis=1)

    demo.create_dataset("absolute_actions", data=absolute_actions)



demo_f.close()




print()
