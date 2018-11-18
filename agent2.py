import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import app
import random


_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4

_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

def _xy_locs(mask):
    y, x = mask.nonzero()
    return list(zip(x, y))

class Agent2(base_agent.BaseAgent):
    def step(self, obs):
        super(Agent2, self).step(obs)
        if _MOVE_SCREEN in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            beacon_center = np.mean(beacon, axis=0).round()
            return actions.FUNCTIONS.Move_screen("now", beacon_center)
        else:
            return actions.FUNCTIONS.select_army("select")

def main(unused_argv):
    agent = Agent2()
    try:
        while True:
            with sc2_env.SC2Env(map_name="MoveToBeacon", 
                                agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True),
                                step_mul=16,
                                game_steps_per_episode=0,
                                visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__== "__main__":
    app.run(main)