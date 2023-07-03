import numpy as np
from numpy import pi, cos, sin

# from spatialmath import SE3


class FrankaFK:
    def __init__(self):
        self.dh_params = [
            [0, 0, 0.333],
            [-pi / 2, 0, 0],
            [pi / 2, 0, 0.316],
            [pi / 2, 0.0825, 0],
            [-pi / 2, -0.0825, 0.384],
            [pi / 2, 0, 0],
            [pi / 2, 0.088, 0.107],
        ]
        self.dof = 7
        # TODO: add Tool offset

    def _construct_dh_matrix(self, joint_idx, q):
        alpha_i, a_i, d_i = self.dh_params[joint_idx]
        q_i = q.flatten()[joint_idx]
        return np.array(
            [
                [cos(q_i), -sin(q_i), 0, a_i],
                [sin(q_i) * cos(alpha_i), cos(q_i) * cos(alpha_i), -sin(alpha_i), -sin(alpha_i) * d_i],
                [sin(q_i) * sin(alpha_i), cos(q_i) * sin(alpha_i), cos(alpha_i), cos(alpha_i) * d_i],
                [0, 0, 0, 1],
            ]
        )

    def get_dh_matrices(self, q):
        # Define Transformation matrix based on DH params
        return [self._construct_dh_matrix(i, q) for i in range(self.dof)]

    def get_joint_poses(self, q, format="SE3"):
        # TODO: add quaternion support
        dh_matrices = self.get_dh_matrices(q)
        X = np.eye(4)
        # the first matrix is identity by default, the reset 1:dof+1 are the joint poses
        joint_poses = np.eye(4)[np.newaxis, ...]
        joint_poses = np.repeat(joint_poses, self.dof+1, axis=0)
        for i in range(0, self.dof):
            X = X @ dh_matrices[i]
            joint_poses[i+1, ...] = X
        return joint_poses

    def get_joint_positions(self, q):
        return self.get_joint_poses(q)[:, :3, 3]

    def get_end_effector_pose(self, q):
        return self.get_joint_poses(q)[-1, ...]


# Test Module

if __name__ == "__main__":
    import roboticstoolbox as rtb
    robot = rtb.models.Panda()
    q = np.array([0, -0.3, 0, -2.2, 0, 2, 0.7854]).reshape(7, 1)

    Te = robot.fkine_all(robot.qr)  # forward kinematics
    print(Te)
    franka_fk = FrankaFK()
    print(np.around(franka_fk.get_joint_poses(q), decimals=4))
    # print(np.around(franka_fk.get_end_effector_pose(robot.qr), decimals=4))
