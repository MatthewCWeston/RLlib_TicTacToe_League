from collections import defaultdict

import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

from algo.modules.CMAPPOActionMaskingTorchRLModule import OBSERVATIONS, ACTION_MASK
from environments.TicTacToe import TTT_WIN_PATHS


class RandHeuristicRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_heuristics = []

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = []
        batch = batch[Columns.OBS]
        if (isinstance(batch, dict)):
          batch=batch[OBSERVATIONS]
        for i, obs in enumerate(batch):
            board = obs.reshape((2,9)) # 2x9 array
            action = -1
            for h in self.sub_heuristics:
                action = h(board)
                if (action != -1):
                    break
            if (action == -1): # Fallback
                mask = board.sum(axis=0)
                action = np.random.choice(np.where(mask==0)[0])
            ret.append(action)

        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError()

    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]

class BlockWinHeuristicRLM(RandHeuristicRLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_heuristics = [self.blockOrWin]

    def blockOrWin(self, board):
        # If either player has two more marks in one line than the other, move in the remaining position
        board_subtracted = (board[0]-board[1])
        for p in TTT_WIN_PATHS: # Win, if you can.
            path_state = board_subtracted[p]
            if (path_state.sum()==2):
                return p[np.where(path_state==0)[0][0]]
        for p in TTT_WIN_PATHS: # Avoid losing, if you must.
            path_state = board_subtracted[p]
            if (path_state.sum()==-2):
                return p[np.where(path_state==0)[0][0]]
        return -1

class PerfectHeuristicRLM(BlockWinHeuristicRLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_heuristics = [self.blockOrWin, self.handleTraps, self.takeCenter]

    def takeCenter(self, board): # Take the center position if it's unoccupied
      return 4 if (board[0][4]+board[1][4]==0) else -1

    def handleTrapsForPlayer(self, board, player):
      # Go through all rows, identify the ones with sum=1
      board_subtracted = (board[player]-board[1-player])
      board_added = board.sum(axis=0)
      viable_paths = []
      for p in TTT_WIN_PATHS: # Get paths with only one dot
          sbp = board_added[p]
          if (sbp.sum() == 1 and board_subtracted[p].sum() == 1):
            viable_paths.append(p[np.where(sbp==0)[0]])
      for i, p in enumerate(viable_paths[:-1]):
          for p2 in viable_paths[(i+1):]:
            u = np.intersect1d(p, p2)
            if (len(u)>0): # create or block a situation where two paths have 2
              return u[0]
      return -1

    def handleTraps(self, board):
      # Setting traps has greater priority than blocking them.
      action = self.handleTrapsForPlayer(board, 0)
      if (action==-1):
          return self.handleTrapsForPlayer(board, 1)
      return action