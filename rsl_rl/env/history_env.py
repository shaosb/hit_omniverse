from .vec_env import VecEnv
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
import torch


class HistoryEnv(RslRlVecEnvWrapper):
    def __init__(self, env: ManagerBasedRLEnv, agent_cfg):
        super().__init__(env)
        self.obs_context_len = agent_cfg["obs_context_len"]
        self.privileged_context_len = agent_cfg["privileged_context_len"]
        self.obs_history_buf = torch.zeros(self.num_envs, self.obs_context_len, self.num_obs, device=self.device, dtype=torch.float)
        self.privileged_history_buf = torch.zeros(self.num_envs, self.privileged_context_len, self.num_privileged_obs, device=self.device, dtype=torch.float)
        self.num_obs = self.num_obs * self.obs_context_len
        self.num_privileged_obs = self.num_privileged_obs * self.privileged_context_len

        # self.original_reset_id = self.env.unwrapped._reset_idx
        # self.env.unwrapped._reset_idx = self._reset_idx.__get__(env.unwrapped)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs, extras = super().get_observations()
        self.obs_history_buf = torch.cat([
            self.obs_history_buf[:, 1:],
            obs.unsqueeze(1)
        ], dim=1)
        return self.obs_history_buf.view(self.num_envs, -1), extras
    
    def get_critic_observations(self, privileged_obs:torch.Tensor) -> torch.Tensor:
        self.privileged_history_buf = torch.cat([
            self.privileged_history_buf[:, 1:],
            privileged_obs.unsqueeze(1)
        ], dim=1)
        return self.privileged_history_buf.view(self.num_envs, -1)

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.obs_history_buf.zero_()
        self.privileged_history_buf.zero_()
        return super().reset()

    # def _reset_idx(self, env_ids):
    #     self.obs_history_buf[env_ids] = 0
    #     self.privileged_history_buf[env_ids] = 0
    #     # return super().unwrapped._reset_idx(env_ids)
    #     return self.original_reset_id(env_ids)

    def step(self, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(action)

        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        if "critic" in obs_dict:
            critic = obs_dict["critic"]
        else:
            critic = obs

        self.obs_history_buf = torch.cat([
            self.obs_history_buf[:, 1:],
            obs.unsqueeze(1)
        ], dim=1)
        self.privileged_history_buf = torch.cat([
            self.privileged_history_buf[:, 1:],
            critic.unsqueeze(1)
        ], dim=1)

        observation = {"policy": self.obs_history_buf.view(self.num_envs, -1), "critic": self.privileged_history_buf.view(self.num_envs, -1)}
        extras["observations"] = observation

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return self.obs_history_buf.view(self.num_envs, -1), rew, dones, extras
