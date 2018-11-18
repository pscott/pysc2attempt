import math
import numpy as np
from pysc2.agent import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000
_BACKGROUND = 0

_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

possible_actions = [
    _NO_OP,
    _SELECT_ARMY,
    _SELECT_POINT,
    _MOVE_SCREEN,
    _MOVE_RAND,
    _MOVE_MIDDLE
    ]

def _xy_locs(mask):
    y, x = mask.nonzero()
    return list(zip(x, y))

class Agent3(base_aget.BaseAgent):
    def __init__(self, load_qt=None, load_st=None):
        super(Agent3, self).__init__()
        self.qtable = QTable(possible_actions, load_qt='agent3_qtable.npy', load_st='agent3_states.npy')

    def step(self, obs):
        super(Agent3, self).step(obs)
        action = self.qtable.get_action(state)
        func = actions.FUNCTIONS.no_op()

        if possible_actions[action] == _NO_OP:
            func = actions.FUNCTIONS.no_op()
        elif state[0] and possible_actions[action] == _MOVE_SCREEN:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            beacon_center = np.mean(beacon, axis=0).round()
            func = actions.FUNCTIONS.Move_screen("now", beacon_center)
            