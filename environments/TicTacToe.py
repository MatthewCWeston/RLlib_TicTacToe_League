import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
)

import numpy as np

from algo.constants import ACTION_MASK, OBSERVATIONS

TTT_WIN_PATHS = np.array([
    [0,1,2],[3,4,5],[6,7,8], # Horizontal
    [0,3,6],[1,4,7],[2,5,8], # Vertical
    [0,4,8],[2,4,6]          # Diagonal
])

class TicTacToe(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.agents = self.possible_agents = ['X','O']
        #"""
        space = gym.spaces.Dict({
            ACTION_MASK: Box(0.0, 1.0, shape=(9,)),
            OBSERVATIONS: Box(0.0, 1.0, (18,), np.float32)
        })
        """
        space = Box(0.0, 1.0, (18,), np.float32) # """
        self.observation_spaces = {
            'X': space,
            'O': space,
        }
        self.action_spaces = {
            'X': Discrete(9),
            'O': Discrete(9),
        }
        self.board = None
        self.current_player = None
    def get_obs(self):
        board = self.board.copy()
        if (self.current_player=='O'):
          board = board[::-1]
        #'''
        return {
            OBSERVATIONS: board.flatten(),
            ACTION_MASK: 1-board.sum(axis=0)
        } #'''
        #return board.flatten()
    def reset(self, *, seed=None, options=None):
        self.board = np.zeros((2,9), dtype=np.float32)
        self.current_player = 'X'
        return {
            self.current_player: self.get_obs(),
        }, {}
    def step(self, action_dict):
        action = action_dict[self.current_player]
        rewards = {self.current_player: 0.0}
        terminateds = {"__all__": False}
        opponent = 'X' if self.current_player=='O' else 'O'
        # Penalize trying to place a piece on an already occupied field.
        action_mask = self.board.sum(axis=0) # 1 -> masked, 0 -> okay
        if action_mask[action] != 0:
            #rewards[self.current_player] -= 0.5
            print("!!! An invalid move was made!")
        else:
            # The current player will always see himself as the first row and his opponent as the second row
            board_ix = 0 if self.current_player=='X' else 1
            self.board[board_ix][action] = 1
            board = self.board[board_ix]
            win_val = [1, 1, 1]
            for p in TTT_WIN_PATHS:
                if (board[p].sum()==3):
                    rewards[self.current_player] = 1.0
                    rewards[opponent] = -1.0
                    # Episode is done and needs to be reset for a new game.
                    terminateds["__all__"] = True
            # The board might also be full w/o any player having won/lost.
            if (self.board.sum()==9) and (terminateds["__all__"]==False):
                terminateds["__all__"] = True
        #display_board(self.board)
        self.current_player = opponent
        return (
            {self.current_player: self.get_obs()},
            rewards,
            terminateds,
            {},
            {},
        )

def display_board(board):
  board = board.reshape((2,9))
  moves = []
  for x, o in zip(board[0], board[1]):
    if (x):
      moves.append('X')
    elif (o):
      moves.append('O')
    else:
      moves.append('_')
  print(moves[:3])
  print(moves[3:6])
  print(moves[6:])
  print('-----')

def convert_board(s):
  s = s.replace('\n','')
  board = np.zeros((2,9))
  for i, c in enumerate(s):
    if (c=='X'):
      board[0][i] = 1
    elif (c=='O'):
      board[1][i] = 1
  return board
  
