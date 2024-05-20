# pipe.py: Implementação do projeto de Inteligência Artificial 2023/2024.


# Grupo 23:
# 107030 Gabriel dos Reis Fonseca Castelo Ferreira
# 106537 Francisco Rodrigues Martins Gonçalves Fernandes

import sys
import numpy as np
import copy
# from search import (
#   Problem,
#   Node,
#   astar_search,
#   breadth_first_tree_search,
#   depth_first_tree_search,
#   greedy_search,
#   recursive_best_first_search,
# )
from search import *

### Global Constants
# Representation of a Piece in 4-bits:
# 1 represents a connection, 0 represents no connection
# Order - up-down-left-right
TOP_MASK  = np.uint8(0b1000)
BOTTOM_MASK = np.uint8(0b0100)
LEFT_MASK   = np.uint8(0b0010)
RIGHT_MASK  = np.uint8(0b0001)
DIRECTIONS = [TOP_MASK, BOTTOM_MASK, LEFT_MASK, RIGHT_MASK]
STR_TO_PIECE: dict = {
  '0':  0b0000,  'FD': 0b0001,  'FE': 0b0010,  
  'LH': 0b0011,  'FB': 0b0100,  'VB': 0b0101,  
  'VE': 0b0110,  'BB': 0b0111,  'FC': 0b1000,  
  'VD': 0b1001,  'VC': 0b1010,  'BC': 0b1011,  
  'LV': 0b1100,  'BD': 0b1101,  'BE': 0b1110,
}
PIECE_TO_STR: list = [
  '', 'FD', 'FE', 'LH', 'FB', 'VB', 'VE', 'BB', 
  'FC', 'VD', 'VC', 'BC', 'LV', 'BD', 'BE']
PIECE_ROTATIONS = [
  [],             # 0b0000 
  [0b0100, 0b1000, 0b0010], # 0b0001
  [0b0100, 0b1000, 0b0001], # 0b0010
  [0b1100],         # 0b0011
  [0b0010, 0b1000, 0b0001], # 0b0100
  [0b0110, 0b1010, 0b1001], # 0b0101
  [0b0101, 0b1010, 0b1001], # 0b0110
  [0b1101, 0b1110, 0b1011], # 0b0111
  [0b0010, 0b0100, 0b0001], # 0b1000
  [0b0101, 0b1010, 0b0110], # 0b1001
  [0b0101, 0b1001, 0b0110], # 0b1010
  [0b1110, 0b0111, 0b1101], # 0b1011
  [0b0011],         # 0b1100
  [0b1110, 0b0111, 0b1011], # 0b1101
  [0b1101, 0b0111, 0b1011], # 0b1110
]

class PipeManiaState:
  """Class that represents a Problem Node's State"""

  state_id = 0

  def __init__(self, board):
    self.board: Board = board
    self.id = PipeManiaState.state_id
    PipeManiaState.state_id += 1

  def __lt__(self, other):
    return self.id < other.id

  def print(self):
    for row in self.board.matrix[1:-1,1:-1]:
      print('\t'.join(map(lambda x: PIECE_TO_STR[x], row)))


class Board:
  """Representação interna de um tabuleiro de PipeMania."""
  def __init__(self, matrix: np.ndarray, moves) -> None:
    self.matrix = matrix
    self.side = matrix.shape[0] - 2
    self.moves = moves
    if moves == None:
      self.prune_first_time()
    else:
      self.move_grid: list = self.matrix[..., np.newaxis].tolist()
      for i, j, x in self.moves:
        self.move_grid[i][j].extend(x)

  def prune_unit(self, visited: list, i, j) -> None:
    """For a given position prune the actions around it"""
    unchanged = True
    if not visited[i][j][0]:
      # Check if top is fixed
      top = self.move_grid[i][j][0] & 0b1000
      for a in self.move_grid[i][j]:
        if a & 0b1000 != top:
          top = None
      if top != None:
        # If top is fixed, force piece on its top to match
        top >>= 1
        self.move_grid[i-1][j] = [a for a in self.move_grid[i-1][j] if a & 0b0100 == top]
        visited[i][j][0] = True
        visited[i-1][j][1] = True
        unchanged = False
    if not visited[i][j][1]:
      # Check if bottom is fixed
      bottom = self.move_grid[i][j][0] & 0b0100
      for a in self.move_grid[i][j]:
        if a & 0b0100 != bottom:
          bottom = None
      if bottom != None:
        # If bottom is fixed, force piece on its bottom to match
        bottom <<= 1
        self.move_grid[i+1][j] = [a for a in self.move_grid[i+1][j] if a & 0b1000 == bottom]
        visited[i][j][1] = True
        visited[i+1][j][0] = True
        unchanged = False
    if not visited[i][j][2]:
      # Check if left is fixed
      left = self.move_grid[i][j][0] & 0b0010
      for a in self.move_grid[i][j]:
        if a & 0b0010 != left:
          left = None
      if left != None:
        # If left is fixed, force piece on its left to match
        left >>= 1
        self.move_grid[i][j-1] = [a for a in self.move_grid[i][j-1] if a & 0b0001 == left]
        visited[i][j][2] = True
        visited[i][j-1][3] = True
        unchanged = False
    if not visited[i][j][3]:
      # Check if right is fixed
      right = self.move_grid[i][j][0] & 0b0001
      for a in self.move_grid[i][j]:
        if a & 0b0001 != right:
          right = None
      if right != None:
        # If right is fixed, force piece on its right to match
        right <<= 1
        self.move_grid[i][j+1] = [a for a in self.move_grid[i][j+1] if a & 0b0010 == right]
        visited[i][j][3] = True
        visited[i][j+1][2] = True
        unchanged = False
    return unchanged
  
  def prune_first_time(self):
    """Infere que estados estão a mais na matrix de ações possíveis"""
    # Create moves matrix, if it does not exist
    self.move_grid = [[[x]+PIECE_ROTATIONS[x] for x in row] for row in self.matrix]
    # Initial pruning
    for i in range(1, self.side+1):
      # Remove actions connecting up on top row
      self.move_grid[1][i] = [action for action in self.move_grid[1][i] if not action & TOP_MASK]
      # Remove actions connecting down on bottom row
      self.move_grid[-2][i] = [action for action in self.move_grid[-2][i] if not action & BOTTOM_MASK]
      # Remove actions connecting left on left col
      self.move_grid[i][1] = [action for action in self.move_grid[i][1] if not action & LEFT_MASK]
      # Remove actions connecting right on right col
      self.move_grid[i][-2] = [action for action in self.move_grid[i][-2] if not action & RIGHT_MASK]
      # Remove actions connecting to dead ends
      for j in range(1, self.side):
        if 0b0001 in self.move_grid[i][j] and 0b0010 in self.move_grid[i][j+1]: # right and left
          self.move_grid[i][j].remove(0b0001)
          self.move_grid[i][j+1].remove(0b0010)
        if 0b0100 in self.move_grid[j][i] and 0b1000 in self.move_grid[j+1][i]: # up and down
          self.move_grid[j][i].remove(0b0100)
          self.move_grid[j+1][i].remove(0b1000)
    # Prune every position
    visited = [[[False, False, False, False] for _ in range(self.side + 2)] for _ in range(self.side + 2)]
    while True:
      unchanged = True
      for i in range(1, self.side+1):
        for j in range(1, self.side+1):
          unchanged &= self.prune_unit(visited, i, j)
      if unchanged:
        break
        
    # Reset matrix state and remove actions that have no alternative
    for i in range(1, self.side+1):
      for j in range(1, self.side+1):
        self.matrix[i,j] = self.move_grid[i][j][0]
    self.moves = [
      (i, j, tuple(self.move_grid[i][j][1:]))
      for i in range(1, self.side+1)
      for j in range(1, self.side+1)
      if len(self.move_grid[i][j][1:]) > 0
    ]
    del self.move_grid

  def prune_moves(self):
    """Infere que estados estão a mais na matrix de ações possíveis"""
    # Prune every position
    visited = [[[False, False, False, False] for _ in range(self.side + 2)] for _ in range(self.side + 2)]
    while True:
      unchanged = True
      for i, j, _ in self.moves:
        if len(self.move_grid[i][j]) == 0:
          continue
        unchanged &= self.prune_unit(visited, i, j)
      if unchanged:
        break

    for i, j, _ in self.moves:
      if len(self.move_grid[i][j]) == 0:
        self.moves = []
        return
      self.matrix[i,j] = self.move_grid[i][j][0]

    self.moves = [
      (i, j, tuple(self.move_grid[i][j][1:])) 
      for i,j,_ in self.moves 
      if len(self.move_grid[i][j][1:]) > 0
    ]
    del self.move_grid

  @staticmethod
  def parse_instance():
    """Lê o test do standard input (stdin) que é passado como argumento
    e retorna uma instância da classe Board.
    """
    matrix = []
    for line in sys.stdin:
      matrix.append([0]+[STR_TO_PIECE[x] for x in line.split()]+[0])
    matrix.insert(0,[0]*len(matrix[0]))
    matrix.append([0]*len(matrix[0]))
    return Board(np.array(matrix), None)


class PipeMania(Problem):
  def __init__(self, board: Board):
    """O construtor especifica o estado inicial."""
    self.initial = PipeManiaState(board)

  def actions(self, state: PipeManiaState):
    """Retorna uma lista de ações que podem ser executadas a
    partir do estado passado como argumento."""
    # Sort moves based on whether 
    # for i, j, ps in sorted(state.board.moves, key=lambda x: len(x[2])):
    state.board.moves.sort(key=lambda x: len(x[2]))
    for i, j, ps in state.board.moves:
      for p in ps:
      # if state.board.matrix[i,j] != p:
        yield (i,j,p)
    del state.board.moves

  def result(self, state: PipeManiaState, action):
    """Retorna o estado resultante de executar a 'action' sobre
    'state' passado como argumento. A ação a executar deve ser uma
    das presentes na lista obtida pela execução de
    self.actions(state)."""
    row, col, piece = action
    new_board = Board(np.copy(state.board.matrix), state.board.moves)
    new_board.matrix[row, col] = piece
    new_board.move_grid[row][col] = [piece]
    new_board.prune_moves()
    return PipeManiaState(new_board)

  def goal_test(self, state: PipeManiaState):
    """Retorna True se e só se o estado passado como argumento é
    um estado objetivo. Deve verificar se todas as posições do tabuleiro
    estão preenchidas de acordo com as regras do problema."""
    m: np.ndarray = state.board.matrix
    # print(m)
    # shifted = m << 1
    # vertical = shifted[:-1,] ^ m[1:,]
    # vertical &= 0b1000
    # if (np.any(vertical)):
    #   return False
    # horizontal = shifted[:,:-1] ^ m[:,1:]
    # horizontal &= 0b0010
    # if np.any(horizontal):
    #   return False
    if (state.h != 0):
      return False
    # DFS that checks if all pieces are connected to (1,1)
    m = m.tolist()
    visited = [[False for _ in range(0, len(m))] for _ in range(0, len(m))]
    frontier = [(1,1)]
    while frontier:
      row, col = frontier.pop()
      if (visited[row][col]):
        continue
      visited[row][col] = True
      cell = m[row][col]
      if cell & 0b1000 and m[row-1][col] & 0b0100:
        frontier.append((row-1, col))
      if cell & 0b0100 and m[row+1][col] & 0b1000:
        frontier.append((row+1, col))
      if cell & 0b0010 and m[row][col-1] & 0b0001:
        frontier.append((row, col-1))
      if cell & 0b0001 and m[row][col+1] & 0b0010:
        frontier.append((row, col+1))
    return np.sum(visited) == self.initial.board.side ** 2

  def h(self, node: Node):
    """Função heuristica utilizada para a procura A*."""
    # DFS to find number of connected pieces to piece (1,1)
    node.state.h = len(node.state.board.moves)
    return len(node.state.board.moves)
    # m = node.state.board.matrix.tolist()
    # visited = [[False for _ in range(0, len(m))] for _ in range(0, len(m))]
    # frontier = [(1,1)]
    # while frontier:
    #   row, col = frontier.pop()
    #   if (visited[row][col]):
    #     continue
    #   visited[row][col] = True
    #   cell = m[row][col]
    #   if cell & 0b1000 and m[row-1][col] & 0b0100:
    #     frontier.append((row-1, col))
    #   if cell & 0b0100 and m[row+1][col] & 0b1000:
    #     frontier.append((row+1, col))
    #   if cell & 0b0010 and m[row][col-1] & 0b0001:
    #     frontier.append((row, col-1))
    #   if cell & 0b0001 and m[row][col+1] & 0b0010:
    #     frontier.append((row, col+1))
    # node.state.board.h = node.state.board.side**2 - np.sum(visited)
    # return node.state.board.h

if __name__ == "__main__":
  prob = PipeMania(Board.parse_instance())
  sol = greedy_search(prob)
  sol.state.print()

