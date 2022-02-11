from rlgym.envs import Match


class MatchCreator:
    def __init__(self, state_setter, reward, obs_builder, action_parser, terminal_conds, game_speed=100):
        self.state_setter = state_setter
        self.reward = reward
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.terminal_conds = terminal_conds
        self.game_speed = game_speed

    def get_match(self):
        return Match(
             self.reward,
             self.terminal_conds,
             self.obs_builder,
             self.action_parser,
             self.state_setter,
             game_speed=self.game_speed,
             tick_skip=8
        )
