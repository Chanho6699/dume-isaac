"""RSL-RL PPO runner cfg for Reach. Initial baseline, not tuned.

Values follow the Isaac Lab v2.3.2 Franka reach agent
(isaaclab_tasks/manager_based/manipulation/reach/config/franka/agents/rsl_rl_ppo_cfg.py).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from dume_isaac.tasks.agent_compat import compat_cfg


@configclass
class SO101ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "so101_reach"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = compat_cfg(
        RslRlPpoActorCriticCfg,
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = compat_cfg(
        RslRlPpoAlgorithmCfg,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
