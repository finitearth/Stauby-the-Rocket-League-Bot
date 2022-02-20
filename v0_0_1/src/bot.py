from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np
from rlgym_compat import GameState
from obs_builder import CustomObsBuilder, PolarObsBuilder
from action_parser import CustomActionParser
from agent import Agent


class Stauby(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.obs_builder = PolarObsBuilder()
        self.action_parser = CustomActionParser()
        self.agent = Agent()

        self.tick_skip = 8
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.prev_score = 0
        self.k = 0
        self.n = 0
        self.in_orange_team = None

    def initialize_agent(self):
        self.game_state = GameState(self.get_field_info())
        self.in_orange_team = True#player.team_num == common_values.ORANGE_TEAM
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time
        ticks_elapsed = delta // 0.008  # Smaller than 1/120 on purpose
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action:
            self.update_action = False
            player = self.game_state.players[self.index]# currently changed to freeplay
            # opponents = [p for p in self.game_state.players if p.team_num != self.team] # for multiple player
            # self.game_state.players = [player, opponents[0]] # for 1v1

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.agent.act(obs)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_controls(self.action)
            self.update_action = True

        score = self.game_state.orange_score if self.in_orange_team else self.game_state.blue_score
        if score != self.prev_score: self._chat()
        self.prev_score = score
        return self.controls

    def update_controls(self, action):
        self.action_parser.parse_actions(action, None)
        self.controls.throttle = action[0] #
        self.controls.steer = action[1] #
        # self.controls.pitch = action[2]
        # self.controls.yaw = action[3]
        # self.controls.roll = action[4]
        # self.controls.jump = action[5] > 0
        # self.controls.boost = action[6] > 0  #
        # self.controls.handbrake = action[7] > 0

    def _chat(self):
        try:
            cms = [25, 52, 55, 12, 47, 45, 44, 37, 64, 24, 12, 17]
            cm = cms[self.k]
            self.send_quick_chat(team_only=False, quick_chat=cm)
            if cm == 12:
                self.send_quick_chat(team_only=False, quick_chat=cm)
                self.send_quick_chat(team_only=False, quick_chat=cm)
                # Chat disabled
            self.k = (self.k + 1) % len/cms

        except Exception as e:
            print(e)

    def handle_quick_chat(self, index, team, quick_chat):
        try:
            if quick_chat in [12, 16]:
                cms = [59, 22, 50, 54, 53]
                cm = cms[self.n]
                self.send_quick_chat(team_only=False, quick_chat=cm)
                self.n = (self.n + 1) % len(cms)
        except Exception as e:
            print(e)
