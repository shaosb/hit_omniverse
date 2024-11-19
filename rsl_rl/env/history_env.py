from .vec_env import VecEnv
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
import torch

class HistoryEnv(RslRlVecEnvWrapper):
    def __init__(self, env:ManagerBasedRLEnv, agent_cfg):
        super().__init__(env)
        self.obs_context_len = agent_cfg["obs_context_len"]
        self.privileged_context_len = agent_cfg["privileged_context_len"]
        self.obs_history_buf = torch.zeros(self.num_envs, self.obs_context_len, self.num_obs, device=self.device, dtype=torch.float)
        self.privileged_history_buf = torch.zeros(self.num_envs, self.privileged_context_len, self.num_privileged_obs, device=self.device, dtype=torch.float)
        self.num_obs = self.num_obs * self.obs_context_len
        self.num_privileged_obs = self.num_privileged_obs * self.privileged_context_len

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
    
    def step(self, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, infos = super().step(action)
        self.obs_history_buf = torch.cat([
            self.obs_history_buf[:, 1:],
            obs.unsqueeze(1)
        ], dim=1)
        if "critic" in infos["observations"]:
            critic_obs = infos["observations"]["critic"]
            temp1 = self.privileged_history_buf[:, 1:]
            temp2 = critic_obs.unsqueeze(1)
            self.privileged_history_buf = torch.cat([
                self.privileged_history_buf[:, 1:],
                critic_obs.unsqueeze(1)
            ], dim=1)
            infos["observations"]["critic"] = self.privileged_history_buf.view(self.num_envs, -1)

        return self.obs_history_buf.view(self.num_envs, -1), rewards, dones, infos
        