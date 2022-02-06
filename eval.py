import rlgym
from stable_baselines3 import PPO
from train import CustomObsBuilder, CustomActionParser

env = rlgym.make(game_speed=1, obs_builder=CustomObsBuilder())
model = PPO("MlpPolicy", env=env, verbose=1)
model = model.load("models/v0_1")
action_parser = CustomActionParser()

while True:
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        action = action_parser.parse_actions(action, None)

        next_obs, reward, done, gameinfo = env.step(action)
        obs = next_obs
