''' GatheringDilemmaMDPClass.py: Contains an implementation of Gathering from
the Deep Mind paper Multi-agent Reinforcement Learning in Sequential Social
Dilemmas. '''

# Python imports.
import random
import numpy as np

# Other imports.
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from simple_rl.tasks.gathering.GatheringStateClass import GatheringState
from simple_rl.tasks.gathering.GatheringStateClass import GatheringAgent

# TODO: remove when State is changed to GatheringState
from simple_rl.mdp.StateClass import State

# Static constants
INITIAL_ORIENTATION = 'NORTH'
ACTIONS = [
    "step_forward",
    "step_backward",
    "step_left",
    "step_right",
    "rotate_left",
    "rotate_right",
    "use_beam",
    "stand_still",
]
ROTATION_MATRICES = {
    'NORTH' : np.matrix([[1, 0],[0, 1]]),
    'SOUTH' : np.matrix([[-1, 0],[0, -1]]),
    'EAST' : np.matrix([[0, -1],[1, 0]]),
    'WEST' : np.matrix([[0, 1],[-1, 0]]),
}
MOVEMENT_VECTOR = {
    'step_forward' : np.matrix([[0], [-1]]),
    'step_backward' : np.matrix([[0], [1]]),
    'step_left' : np.matrix([[-1], [0]]),
    'step_right' : np.matrix([[1], [0]]),
}
ROTATE_LEFT = {
    'NORTH' : 'WEST',
    'SOUTH' : 'EAST',
    'EAST' : 'NORTH',
    'WEST' : 'SOUTH',
}
ROTATE_RIGHT = {
    'NORTH' : 'EAST',
    'SOUTH' : 'WEST',
    'EAST' : 'SOUTH',
    'WEST' : 'NORTH',
}

class GatheringMDP(MarkovGameMDP):

    def __init__(
        self,
        gamma,
        possible_apple_locations,
        N_apples,
        N_tagged,
        dim=(35, 13),
    ):
        self.gamma, self.N_apples, self.N_tagged = gamma, N_apples, N_tagged
        self.x_dim, self.y_dim = dim[0], dim[1]

        agent1 = GatheringAgent(31, 6, False, INITIAL_ORIENTATION, 0, 0)
        agent2 = GatheringAgent(32, 5, False, INITIAL_ORIENTATION, 0, 0)
        idx = np.array(possible_apple_locations)
        initial_apple_times = dict()
        for loc in possible_apple_locations:
            initial_apple_times[loc] = 1 #random.randint(2, 7) # random spawn times for apples

        # print(idx)

        initial_apple_locations = np.zeros(shape=[self.x_dim, self.y_dim],
            dtype=np.int32)
        initial_apple_locations[idx[:, 0], idx[:, 1]] = 1
        # print(initial_apple_locations)

        MarkovGameMDP.__init__(
            self,
            ACTIONS,
            self._transition_func,
            self._reward_func,
            init_state=GatheringState(agent1, agent2, initial_apple_locations, initial_apple_times),
        )

    def _reward_func(self, state, action_dict):
        # Repeat computations above & update player location if they moved.
        agent_a, agent_b = state.agent1, state.agent2
        agent_a_name, agent_b_name = action_dict.keys()[0], action_dict.keys()[1]
        # print action_dict
        # print state
        reward_dict = {}
        if state.apple_locations[agent_a.x, agent_a.y] == 1:
            reward_dict[agent_a_name] = 1
        else:
            reward_dict[agent_a_name] = 0
        if state.apple_locations[agent_b.x, agent_b.y] == 1:
            reward_dict[agent_b_name] = 1
        else:
            reward_dict[agent_b_name] = 0
        return reward_dict

    def _transition_func(self, state, action_dict):
        # Repeat computations above & update player location if they moved.
        agent_a_name, agent_b_name = action_dict.keys()[0], action_dict.keys()[1]
        action_a, action_b = action_dict[agent_a_name], action_dict[agent_b_name]

        agent_a, agent_b = state.agent1, state.agent2

        ## we should be creating a new object based on the old one and returning that
        ## but maintain old agents
        newState = state.generate_next_state()
        self._update_apples(newState)

        # Two hits leads to being frozen, hits reset after -- not consecutive
        if agent_a.frozen_time_remaining > 0:
            newState.agent1.frozen_time_remaining -= 1
            if newState.agent1.frozen_time_remaining > 0:
                action_a = None
        if agent_b.frozen_time_remaining > 0:
            agent_b.frozen_time_remaining -= 1
            if agent_b.frozen_time_remaining > 0:
                action_b = None

        if action_a == 'use_beam' and agent_b.frozen_time_remaining == 0:
            self._is_hit_by_beam(agent_a, agent_b)
        if action_b == 'use_beam' and agent_a.frozen_time_remaining == 0:
            self._is_hit_by_beam(agent_b, agent_a)

        # Check if they are frozen again
        if agent_a.frozen_time_remaining > 0:
            action_a = None
        if agent_b.frozen_time_remaining > 0:
            action_b = None

        a_x, a_y = agent_a.x, agent_a.y
        if self._can_perform_move(agent_a, action_a):
            a_x, a_y = self._get_next_location(agent_a, action_a)
        b_x, b_y = agent_b.x, agent_b.y
        if self._can_perform_move(agent_b, action_b):
            b_x, b_y = self._get_next_location(agent_b, action_b)

        if a_x == b_x and a_y == b_y:
            if (agent_a.x != a_x or agent_a.y != a_y) and (agent_b.x != b_x or agent_b.y != b_y):
                # 50 / 50 chance when both moving into same space
                    if random.random() > 0.5:
                        agent_a.x, agent_a.y = a_x, a_y
                    else:
                        agent_b.x, agent_b.y = b_x, b_y
            else:
                a_x, a_y = agent_a.x, agent_a.y
                b_x, b_y = agent_b.x, agent_b.y
        # handle swapping locations
        elif a_x == agent_b.x and a_y == agent_b.y and b_x == agent_a.x and b_y == agent_a.y:
            a_x, a_y = agent_a.x, agent_a.y
            b_x, b_y = agent_b.x, agent_b.y

        agent_a.x, agent_a.y = a_x, a_y
        agent_b.x, agent_b.y = b_x, b_y

        for agent, act in [(agent_a, action_a), (agent_b, action_b)]:
            if act == 'rotate_left':
                agent.orientation = ROTATE_LEFT[agent.orientation]
            elif act == 'rotate_right':
                agent.orientation = ROTATE_RIGHT[agent.orientation]

        return newState

    def _can_perform_move(self, agent, action):
        if action == None:
            return False
        if not action.startswith('step'):
            return True

        final_pos_x, final_pos_y = self._get_next_location(agent, action)
        # print final_pos_x
        # print final_pos_y
        return final_pos_x > 0 and \
                final_pos_x < self.x_dim - 1 and \
                final_pos_y > 0 and \
                final_pos_y < self.y_dim - 1

    def _get_next_location(self, agent, action):
        if not action.startswith('step'):
            return agent.x, agent.y
        movement = np.dot(
            ROTATION_MATRICES[agent.orientation],
            MOVEMENT_VECTOR[action],
        )
        return agent.x + movement[0][0], agent.y + movement[1][0]

    # Generate apples based on parameters and pick them up
    ## apples appear after people have moved and not where people are located
    def _update_apples(self, state):
        # iterate through apple Locations
        for apple in state.apple_times.keys():
            apple_x, apple_y = apple[0], apple[1]

            # if it is greater than 1, lower it
            if state.apple_times[apple] > 1:
                state.apple_times[apple] -= 1
            # if it is 1, check to see if a player is there
            elif state.apple_times[apple] == 1:
                # generate an apple
                state.apple_times[apple] = 0
                state.apple_locations[apple_x, apple_y] = 1
            elif state.apple_times[apple] == 0:
                # TODO: Figure out why the below assert statement exists
                # assert state.apple_locations[apple_x, apple_y] == 1, 'Apples not generated properly'
                # if a player is there, remove the apple from the location
                #     and increase the apple time by N_apples
                if (state.agent1.x == apple_x and state.agent1.y == apple_y) \
                    or (state.agent2.x == apple_x and state.agent2.y == apple_y):
                    state.apple_locations[apple_x, apple_y] = 0
                    state.apple_times[apple] = self.N_apples - 1
        return

    def _is_hit_by_beam(self, target, beamer):
        if beamer.orientation == 'NORTH' and target.y == beamer.y and target.x < beamer.x:
            if target.hits == 0:
                target.hits += 1
            else:
                target.frozen_time_remaining = self.N_tagged
                target.hits = 0
            return
        elif beamer.orientation == 'SOUTH' and target.y == beamer.y and target.x > beamer.x:
            if target.hits == 0:
                target.hits += 1
            else:
                target.frozen_time_remaining = self.N_tagged
                target.hits = 0
            return
        elif beamer.orientation == 'EAST' and target.y > beamer.y and target.x == beamer.x:
            if target.hits == 0:
                target.hits += 1
            else:
                target.frozen_time_remaining = self.N_tagged
                target.hits = 0
            return
        elif beamer.orientation == 'WEST' and target.y < beamer.y and target.x == beamer.x:
            if target.hits == 0:
                target.hits += 1
            else:
                target.frozen_time_remaining = self.N_tagged
                target.hits = 0
            return

    def __str__(self):
        return "gathering_game"
