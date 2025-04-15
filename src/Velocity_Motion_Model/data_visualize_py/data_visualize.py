import sys
sys.path.append("C:/Users/guhan/Desktop/Github_Repo/SLAM_For_Robotics/build/Release/")

import velocity_motion_model
import matplotlib.pyplot as plt
import numpy as np

#rotation_1
angular_noise = velocity_motion_model.Noise(0.001, 0.001)
velocity_noise = velocity_motion_model.Noise(0.001, 0.001)
#rotation_2
rotational_noise = velocity_motion_model.Noise(0.001, 0.001)

del_time = 1

motion_model = velocity_motion_model.VelocityMotionModel(velocity_noise, angular_noise, rotational_noise, del_time)

initial_pose = velocity_motion_model.Pose(0.0, 0.0, 0.0)

control_command = velocity_motion_model.ControlCommand(3.0, 0.1)

# temp = motion_model.sample_motion(initial_pose, control_command)
# print(temp.x, temp.y, temp.theta, motion_model.get_posterior_probability(initial_pose, temp, control_command))

probs = []
poses = []

for range in range(500):
    temp = motion_model.sample_motion(initial_pose, control_command)
    poses.append(temp)
    probs.append(motion_model.get_posterior_probability(initial_pose, temp, control_command))
    # print(temp.x, temp.y, temp.theta, probs[-1])

probs = np.array(probs)
probs = 1 / probs
probs = probs / np.max(probs)

plt.scatter(initial_pose.x, initial_pose.y)
for pose, prob in zip(poses, probs):
    plt.scatter(pose.x, pose.y, c='r', alpha=prob)
plt.show()

# plt.plot([initial_pose.x, temp.x], [initial_pose.y, temp.y])
# plt.show()