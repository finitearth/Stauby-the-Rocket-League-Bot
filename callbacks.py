import os

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

import wandb
from stable_baselines3.common.logger import make_output_format, KVWriter, Logger


class CustomLogger(Logger):
    def __init__(self):
        super().__init__("", [make_output_format("stdout", "./logs", "")])

    def dump(self, step=0):
        self.output_formats[0].write(self.name_to_value, self.name_to_excluded, step)
        wandb.log(self.name_to_value)
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()


class CustomEvalCallback(EvalCallback):
    def __init__(self, env=None, **kwargs):
        super().__init__(**kwargs)
        self.env = env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                sync_envs_normalization(self.training_env, self.eval_env)
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=False,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")

                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.env.save(os.path.join(self.best_model_save_path, "best_model_env"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

