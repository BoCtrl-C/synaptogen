import imageio
import numpy as np
import os
from gymnasium.utils.save_video import save_video
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import wandb


GIF_DIR = 'gifs'
VIDEO_DIR = 'videos'


################################################################################
# GIF/video recording
################################################################################

class GIFRecorderCallback(BaseCallback):
    def __init__(self, env_id):
        super().__init__()

        import gymnasium as gym
        self.env = gym.make(env_id, render_mode='rgb_array')
    
    def _on_step(self):
        try:
            images = []

            obs, _ = self.env.reset()
            img = self.env.render()
            
            terminated, truncated = False, False
            while not terminated or not truncated:
                images.append(img)
                action, _ = self.model.predict(obs)
                obs, _, terminated, truncated, _ = self.env.step(action)
                img = self.env.render()

            imageio.mimsave(
                os.path.join(GIF_DIR, 'rl-gif-episode-0.gif'),
                [np.array(img) for i, img in enumerate(images) if i%2 == 0],
                fps=16
            )
            wandb.log({'example':
                wandb.Video(os.path.join(GIF_DIR, 'rl-gif-episode-0.gif'))})
        except: pass

class VideoRecorderCallback(BaseCallback):
    def __init__(self, env_id):
        super().__init__()

        import gymnasium as gym
        self.env = gym.make(env_id, render_mode='rgb_array_list')
    
    def _on_step(self):
        obs, _ = self.env.reset()
        while True:
            action, _ = self.model.predict(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)

            if terminated or truncated:
                save_video(
                    self.env.render(),
                    VIDEO_DIR,
                    fps=self.env.metadata['render_fps'],
                )
                wandb.log({'example':
                    wandb.Video(os.path.join(
                        VIDEO_DIR,
                        'rl-video-episode-0.mp4'
                    ))})
                break

def record_video(env_name, model, video_folder=VIDEO_DIR, **kwargs):
    import gymnasium
    env = gymnasium.make(env_name, render_mode='rgb_array_list')

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            save_video(
                env.render(),
                video_folder=video_folder,
                fps=env.metadata['render_fps'],
                **kwargs
            )
            break

################################################################################
# SNES optimization
################################################################################

def s_to_h_min_s(t):
    """Converts the given time in [s] into a [h]:[min]:[s] format.
    """
    
    h = t//3600
    t = t%3600

    min_ = t//60
    s = t%60
    
    return f'{h:.0f}h:{min_:.0f}min:{s:.0f}s'

################################################################################
# W&B
################################################################################

def get_run_data(project_path, sweep_ids):
    """Returns a dataframe containing all run data from the given sweeps. Sweeps are provided as a dictionary of environment name-sweep ID pairs.
    """
    
    api = wandb.Api()

    # generate a dataframe with data from sweeps
    runs = []
    for env_name, sweep_id in sweep_ids.items():
        sweep = api.sweep(os.path.join(project_path, 'sweeps', sweep_id))

        for run in sweep.runs:
            row = {}
            row.update(run.config), row.update(run.summary)
            row['name'] = run.name
            row['run_id'] = run.id
            row['env_name'] = env_name
            runs += [row]

    return pd.DataFrame(runs)