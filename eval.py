import rlgym
from stable_baselines3 import PPO

env = rlgym.make(game_speed=1)
model = PPO("MlpPolicy", env=env, verbose=1)
model = model.load("models/v0")

while True:
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)

        next_obs, reward, done, gameinfo = env.step(action)
        obs = next_obs