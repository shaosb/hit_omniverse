from collections import deque
import torch
import numpy as np
import math
import os

from omni.isaac.lab.envs import ManagerBasedRLEnv
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.sensors import ContactSensor

from hit_omniverse.utils.helper import setup_config

config = setup_config(os.environ.get("CONFIG"))
# training_config = setup_config(os.environ.get("TRAINING_CONFIG"))["runner"]

def constant_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    # v_x, v_y, ang_x
    return torch.tensor([config["VELOCITY"]], device=env.device).repeat(env.num_envs, 1)

def copysign(a, b):
    # type: (float, torch.Tensor) -> torch.Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


def get_euler_xyz_tensor(env: ManagerBasedRLEnv): # 三维朝向
    quat = mdp.root_quat_w(env)
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


def base_yaw_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)


def base_up_proj(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b

    return base_up_vec[:, 2].unsqueeze(-1)


def base_heading_proj(
    env: ManagerBasedRLEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    to_target_pos[:, 2] = 0.0
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_rotate(asset.data.root_quat_w, asset.data.forward_vec_b)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)


def base_angle_to_target(
    env: ManagerBasedRLEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    return angle_to_target.unsqueeze(-1)


def base_up_proj(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b

    return base_up_vec[:, 2].unsqueeze(-1)


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()

def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials

def root_height_over_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height

def get_contact_sensor_data(env: ManagerBasedRLEnv, bodies) -> torch.Tensor:
    contact_sensor = env.scene["contact_sensor"]

    return contact_sensor.data.net_forces_w[:,contact_sensor.find_bodies(bodies)[0][0],2:3]

def Base_height_penalty(env: ManagerBasedRLEnv,threshold:float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    base_pos_z = mdp.base_pos_z(env,asset_cfg)
    result = (base_pos_z-threshold)**2
    result = result.squeeze(0)
    # print("result",result)
    return result

class Velocity_rate_penalty(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.vel_deque = deque(maxlen=3)
        for i in range(3):
            ex_zeros = torch.zeros((1,3), device=env.device)
            self.vel_deque.append(ex_zeros)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

        self.vel_deque.append(mdp.base_lin_vel(env)) #tensor([[ 0.0009,  0.0112, -0.0003]], device='cuda:0')
        # print("self.vel_deque",self.vel_deque)
        sum1 = (self.vel_deque[2] - self.vel_deque[1]).sum()
        sum2 = (self.vel_deque[2] - 2.0 * self.vel_deque[1] + self.vel_deque[0]).sum()
        # print("sum1",sum1)
        r14 = (-0.001 * sum1**2 + sum2**2)
        return r14

def high_foot_contact_penalty(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    # print(asset.data.body_names)
    # ['base_link', 'left_leg_link1', 'right_leg_link1', 'waist_link1', 'left_leg_link2', 'right_leg_link2', 'body',
    #  'left_leg_link3', 'right_leg_link3', 'left_arm_link1', 'right_arm_link1', 'left_leg_link4', 'right_leg_link4',
    #  'left_arm_link2', 'right_arm_link2', 'left_leg_link5', 'right_leg_link5', 'left_arm_link3', 'right_arm_link3',
    #  'left_leg_link6', 'right_leg_link6', 'left_arm_link4', 'right_arm_link4']
    # left_asset.data.body_pos_w
    return asset.data.body_pos_w

# def head_to_base_projection_penalty(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     asset: RigidObject = env.scene[asset_cfg.name]
#     x_head =


def joint_pos_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    This is computed as a sum of the absolute value of the difference between the joint position and the reference motion.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = math_utils.wrap_to_pi(asset.data.joint_pos)
    # target = mdp.generated_commands(env, "dataset")["dof_pos"]
    target = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device=env.device).repeat(env.num_envs, 1)
    return torch.exp(-torch.sum(torch.abs(joint_pos - target), dim=1))


def joint_upper_pos_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    This is computed as a sum of the absolute value of the difference between the joint position and the reference motion.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    upper_body = [2, 5, 8, 9, 12, 13, 16, 17, 20, 21]
    joint_pos = asset.data.joint_pos[:, upper_body]
    target = mdp.generated_commands(env, "dataset")["dof_pos"][:, upper_body]
    # return torch.sum(torch.abs(joint_pos - target), dim=1)
    return -5 * torch.mean(torch.abs(joint_pos - target), dim=1)


def joint_lower_pos_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    This is computed as a sum of the absolute value of the difference between the joint position and the reference motion.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    lower_body = [0, 1, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19]
    joint_pos = asset.data.joint_pos[:, lower_body]
    target = mdp.generated_commands(env, "dataset")["dof_pos"][:, lower_body]
    # return torch.sum(torch.abs(joint_pos - target), dim=1)
    return -5 * torch.mean(torch.abs(joint_pos - target), dim=1)


def target_xy_velocities(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    command_velocity = torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)
    residual = asset.data.root_lin_vel_b - command_velocity
    return torch.sum(torch.exp(torch.square(residual - command_velocity)), dim=1)


def feet_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    foot_pos = asset.data.body_state_w[:, asset.find_bodies(["link_l_foot", "link_r_foot"])[0], :2]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    fd = 0.2
    max_df = 0.5
    d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
    d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def foot_slip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    contact = get_contact_sensor_data(env, ["l_foot", "r_foot"]) > 5.
    foot_speed_norm = torch.norm(asset.data.body_state_w[:, asset.find_bodies(["link_l_foot", "link_r_foot"])[0], 10:12], dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)


def track_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    commands = constant_commands(env)
    lin_vel_error = torch.norm(
        commands[:, :3] - asset.data.root_lin_vel_w[:, :3], dim=1)
    lin_vel_error_exp = torch.exp(-lin_vel_error)

    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.abs(
        commands[:, 3] - asset.data.root_ang_vel_w[:, 3])
    ang_vel_error_exp = torch.exp(-ang_vel_error)

    linear_error = 0.2 * (lin_vel_error + ang_vel_error)

    return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error


def track_lin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    commands = constant_commands(env)

    lin_vel_error = torch.sum(torch.abs(
        commands[:, :3] - asset.data.root_lin_vel_w[:, :]), dim=1)
    return torch.exp(-lin_vel_error)
    # return lin_vel_error

def track_ang(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    commands = constant_commands(env)

    lin_ang_error = torch.sum(torch.abs(
        commands[:, 3:6] - asset.data.root_ang_vel_w[:, :]), dim=1)
    return torch.exp(-0.01 * lin_ang_error)
    # return lin_ang_error

def track_lin_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    commands = constant_commands(env)

    lin_x_error = torch.sum(torch.abs(
        commands[:, 0] - asset.data.root_lin_vel_w[:, 0]))
    return torch.exp(-0.001 * lin_x_error)

def reward_feet_contact_force(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    contact_sensor = env.scene["contact_sensor"]
    net_contact_forces = contact_sensor.data.net_forces_w

    max_values = torch.max(net_contact_forces, dim=2)[0].max(dim=1).values
    min_values = torch.max(net_contact_forces, dim=2)[0].min(dim=1).values
    condition = (max_values > 400) & (min_values < 10)
    result = condition.int()

    return result

def torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, :]

    return torch.exp(-0.001 * torch.sum(torch.abs(torques), dim=1))

def reward_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = math_utils.wrap_to_pi(asset.data.joint_pos)
    # pos_target = mdp.generated_commands(env, "sine")
    pos_target = mdp.generated_commands(env, "dataset")["dof_pos"]
    # diff = joint_pos - pos_target
    # diff = diff[:,[2,3,4,8,9,10]]
    diff = torch.sum(torch.square(joint_pos - pos_target), dim=1)
    return diff
    # return torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)


def get_gait_phase(env: ManagerBasedRLEnv):
    phase = env.command_manager.get_term("sine")._get_phase(None)

    sin_pos = torch.sin(2 * torch.pi * phase)

    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    # left foot stance
    stance_mask[:, 0] = sin_pos >= 0
    # right foot stance
    stance_mask[:, 1] = sin_pos < 0
    # Double support phase
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    return stance_mask


def reward_feet_contact_number(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    contact = get_contact_sensor_data(env, ["l_foot", "r_foot"]) > 5.
    stance_mask = get_gait_phase(env)
    reward = torch.where(contact == stance_mask, 1, -0.3)
    return torch.mean(reward, dim=1)

def reward_action_smooth(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    action_smooth = mdp.action_rate_l2(env)
    return torch.exp(-0.1 * action_smooth)


def reward_feet_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    contact = get_contact_sensor_data(env, ["link_l_foot", "link_r_foot"]) > 5.

    # Get the z-position of the feet and compute the change in z-position
    feet_z = asset.data.body_pos_w[:, asset.find_bodies(["link_l_foot", "link_r_foot"])[0], 2] - 0.05
    delta_z = feet_z - env.last_feet_z
    env.feet_height += delta_z
    env.last_feet_z = feet_z

    # Compute swing mask
    swing_mask = 1 - get_gait_phase(env)

    # feet height should be closed to target feet height at the peak
    rew_pos = torch.abs(env.feet_height - 0.06) < 0.01
    rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    env.feet_height *= ~contact
    return rew_pos


def reward_feet_air_time(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    contact = get_contact_sensor_data(env, ["link_l_foot", "link_r_foot"]) > 5.
    stance_mask = get_gait_phase(env)
    contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), env.last_contacts)
    env.last_contacts = contact
    first_contact = (env.feet_air_time > 0.) * contact_filt
    env.feet_air_time += 0.005
    air_time = env.feet_air_time.clamp(0, 0.5) * first_contact
    env.feet_air_time *= ~contact_filt
    return air_time.sum(dim=1)


def joint_vel_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    This is computed as a sum of the absolute value of the difference between the joint position and the reference motion.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel
    target = mdp.generated_commands(env, "dataset")["dof_vel"]
    return torch.exp(-0.1 * torch.sum(torch.square(joint_vel - target), dim=1))


def dataset_dof_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return mdp.generated_commands(env, "dataset")["dof_pos"]


def dataset_dof_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return mdp.generated_commands(env, "dataset")["dof_vel"]


def world_xyz(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    target = mdp.generated_commands(env, "dataset")["robot_world_xyz"]
    asset: Articulation = env.scene[asset_cfg.name]
    world_xyz = asset.data.root_pos_w

    world_xyz_error = torch.exp(-torch.sum(torch.square(target - world_xyz), dim=1))

    return world_xyz_error

def dataset_world_xyz(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return mdp.generated_commands(env, "dataset")["robot_world_xyz"]

def dataset_world_rpy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return mdp.generated_commands(env, "dataset")["robot_world_rpy"]

def reset_robot_position(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    pos_target = mdp.generated_commands(env, "dataset")["dof_pos"]
    default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(pos_target[env_ids], default_joint_vel, env_ids=env_ids)

def track_yaw_row(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    target = torch.tensor([[0, 0]], device=env.device).repeat(env.num_envs, 1)

    return torch.exp(-torch.sum(torch.abs(target - base_yaw_roll(env, asset_cfg)), dim=1))

def joint_toqrue(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.applied_torque[:, :]


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())