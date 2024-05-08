# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 23:
# 107030 Gabriel dos Reis Fonseca Castelo Ferreira
# 106537 Francisco Rodrigues Martins Gonçalves Fernandes

import sys
import numpy as np
from search import *
# (
#   Problem,
#   Node,
#   astar_search,
#   breadth_first_tree_search,
#   depth_first_tree_search,
#   greedy_search,
#   recursive_best_first_search,
# )

# Piece
# 4-bits up bottom left right
strToPiece: dict = {
  b'0':  np.uint8(0b0000),
  b'FD': np.uint8(0b0001),
  b'FE': np.uint8(0b0010),
  b'LH': np.uint8(0b0011),
  b'FB': np.uint8(0b0100),
  b'VB': np.uint8(0b0101),
  b'VE': np.uint8(0b0110),
  b'BB': np.uint8(0b0111),
  b'FC': np.uint8(0b1000),
  b'VD': np.uint8(0b1001),
  b'VC': np.uint8(0b1010),
  b'BC': np.uint8(0b1011),
  b'LV': np.uint8(0b1100),
  b'BD': np.uint8(0b1101),
  b'BE': np.uint8(0b1110),
}

pieceToStr: list = ['', 'FD', 'FE', 'LH', 'FB', 'VB', 'VE', 'BB', 'FC', 'VD', 'VC', 'BC', 'LV', 'BD', 'BE']
pieceToAction = [
                  [], # 0b0000 
                  [np.uint8(0b0100), np.uint8(0b1000), np.uint8(0b0010)], # 0b0001
                  [np.uint8(0b0100), np.uint8(0b1000), np.uint8(0b0001)], # 0b0010
                  [np.uint8(0b1100)], # 0b0011
                  [np.uint8(0b0010), np.uint8(0b1000), np.uint8(0b0001)], # 0b0100
                  [np.uint8(0b0110), np.uint8(0b1010), np.uint8(0b1001)], # 0b0101
                  [np.uint8(0b0101), np.uint8(0b1010), np.uint8(0b1001)], # 0b0110
                  [np.uint8(0b1101), np.uint8(0b1110), np.uint8(0b1011)], # 0b0111
                  [np.uint8(0b0010), np.uint8(0b0100), np.uint8(0b0001)], # 0b1000
                  [np.uint8(0b0101), np.uint8(0b1010), np.uint8(0b0110)], # 0b1001
                  [np.uint8(0b0101), np.uint8(0b1001), np.uint8(0b0110)], # 0b1010
                  [np.uint8(0b1110), np.uint8(0b0111), np.uint8(0b1101)], # 0b1011
                  [np.uint8(0b0011)], # 0b1100
                  [np.uint8(0b1110), np.uint8(0b0111), np.uint8(0b1011)], # 0b1101
                  [np.uint8(0b1101), np.uint8(0b0111), np.uint8(0b1011)], # 0b1110
                ]

class PipeManiaState:
  state_id = 0

  def __init__(self, board, depth):
    self.board = board
    self.id = PipeManiaState.state_id
    PipeManiaState.state_id += 1
    self.depth=depth

  def __lt__(self, other):
    return self.id < other.id

  # TODO: outros metodos da classe


class Board:
  """Representação interna de um tabuleiro de PipeMania."""
  def __init__(self, matrix: np.ndarray) -> None:
    self.matrix: np.ndarray = matrix
    self.side = matrix.shape[0] - 2 # range() é exclusivo no ultimo elemento

  @staticmethod
  def parse_instance():
    """Lê o test do standard input (stdin) que é passado como argumento
    e retorna uma instância da classe Board.
    """
    output = np.pad(np.genfromtxt(sys.stdin, dtype='S2'), 1)
    matrix = np.ndarray(output.shape, dtype='uint8')
    for k in strToPiece:
        matrix[output == k] = strToPiece[k]
    return Board(matrix)

  # TODO: outros metodos da classe


class PipeMania(Problem):
  
  def __init__(self, board: Board):
    """O construtor especifica o estado inicial."""
    self.initial = PipeManiaState(board, 0)
    # Average number of connections per piece
    self.average = np.sum(np.unpackbits(self.initial.board.matrix)) / self.initial.board.side
    self.moves = [[[x]+pieceToAction[x] for x in row] for row in board.matrix]
    for i in range(1, board.side+1):
      # Remove actions connecting up on top row
      self.moves[1][i] = [action for action in self.moves[1][i] if not action & 0b1000]
      # Remove actions connecting down on bottom row
      self.moves[-2][i] = [action for action in self.moves[-2][i] if not action & 0b0100]
      # Remove actions connecting left on left col
      self.moves[i][1] = [action for action in self.moves[i][1] if not action & 0b0010]
      # Remove actions connecting right on right col
      self.moves[i][-2] = [action for action in self.moves[i][-2] if not action & 0b0001]
      # Remove actions connecting to dead ends
      for j in range(1, board.side):
        if 0b0001 in self.moves[i][j] and 0b0010 in self.moves[i][j+1]: # right and left
          self.moves[i][j].remove(0b0001)
          self.moves[i][j+1].remove(0b0010)
        if 0b0100 in self.moves[j][i] and 0b1000 in self.moves[j+1][i]: # up and down
          self.moves[j][i].remove(0b0100)
          self.moves[j+1][i].remove(0b1000)
    visited = [[[False, False, False, False] for _ in range(board.side + 2)] for _ in range(board.side + 2)]
    while True:
      unchanged = True
      for i in range(1, board.side+1):
        for j in range(1, board.side+1):
          if not visited[i][j][0]:
            # Check if top is fixed
            top = self.moves[i][j][0] & 0b1000
            for a in self.moves[i][j]:
              if a & 0b1000 != top:
                top = None
            if top != None:
              # If top is fixed, force piece on its top to match
              self.moves[i-1][j] = [a for a in self.moves[i-1][j] if a & 0b0100 == (self.moves[i][j][0] & 0b1000) >> 1]
              visited[i][j][0] = True
              visited[i-1][j][1] = True
              unchanged = False
          if not visited[i][j][1]:
            # Check if bottom is fixed
            bottom = self.moves[i][j][0] & 0b0100
            for a in self.moves[i][j]:
              if a & 0b0100 != bottom:
                bottom = None
            if bottom != None:
              self.moves[i+1][j] = [a for a in self.moves[i+1][j] if a & 0b1000 == (self.moves[i][j][0] & 0b0100) << 1]
              visited[i][j][1] = True
              visited[i+1][j][0] = True
              unchanged = False
          if not visited[i][j][2]:
            # Check if left is fixed
            left = self.moves[i][j][0] & 0b0010
            for a in self.moves[i][j]:
              if a & 0b0010 != left:
                left = None
            if left != None:
              self.moves[i][j-1] = [a for a in self.moves[i][j-1] if a & 0b0001 == left >> 1]
              visited[i][j][2] = True
              visited[i][j-1][3] = True
              unchanged = False
          if not visited[i][j][3]:
            # Check if right is fixed
            right = self.moves[i][j][0] & 0b0001
            for a in self.moves[i][j]:
              if a & 0b0001 != right:
                right = None
            if right != None:
              self.moves[i][j+1] = [a for a in self.moves[i][j+1] if a & 0b0010 == right << 1]
              visited[i][j][3] = True
              visited[i][j+1][2] = True
              unchanged = False
      if unchanged:
        break

    for i in range(1, board.side+1):
      for j in range(1, board.side+1):
        self.initial.board.matrix[i,j] = self.moves[i][j][0]
        if len(self.moves[i][j]) == 1:
          self.moves[i][j] = []
      #   print(self.moves[i][j], end=" ")
      # print("")

  


  def actions(self, state: PipeManiaState):
    """Retorna uma lista ou iterador de ações que podem ser executadas a
    partir do estado passado como argumento."""
    _row = state.depth // (state.board.side)
    _col = state.depth % (state.board.side)
    # if (_row >= state.board.side):
    #   return
    # for action in self.moves[row+1][col+1]:
    #   yield (row+1, col+1, action)

    for row in range(1, state.board.side+1):
      for col in range(1, state.board.side+1):
        piece = state.board.matrix[row, col]
        for action in self.moves[row][col]:
          if action != piece:
            yield (row, col, action)
    # for row in range(1, 1+_row):
    #   for col in range(1, 1+_col):
    #     piece = state.board.matrix[row, col]
    #     for action in self.moves[row][col]:
    #       if action != piece:
    #         yield (row, col, action)

  def result(self, state: PipeManiaState, action):
    """Retorna o estado resultante de executar a 'action' sobre
    'state' passado como argumento. A ação a executar deve ser uma
    das presentes na lista obtida pela execução de
    self.actions(state)."""
    row, col, piece = action
    new_board = np.copy(state.board.matrix)
    new_board[row, col] = piece
    # print(action)
    return PipeManiaState(Board(new_board), depth=state.depth+1)

  def goal_test(self, state: PipeManiaState):
    """Retorna True se e só se o estado passado como argumento é
    um estado objetivo. Deve verificar se todas as posições do tabuleiro
    estão preenchidas de acordo com as regras do problema."""
    m: np.ndarray = state.board.matrix
    # print(m)
    shifted = m << 1
    vertical = shifted[:-1,] ^ m[1:,]
    vertical &= 0b1000
    if (np.any(vertical)):
      return False
    horizontal = shifted[:,:-1] ^ m[:,1:]
    horizontal &= 0b0010
    if np.any(horizontal):
      return False
    # return True
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
    m = node.state.board.matrix.tolist()
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
    m = node.state.board.matrix
    shifted = m << 1
    vertical = shifted[:-1,] ^ m[1:,]
    vertical &= 0b1000
    horizontal = shifted[:,:-1] ^ m[:,1:]
    horizontal &= 0b0010
    leaks = (np.sum(vertical) / 8 + np.sum(horizontal) / 4) * self.average
    # return leaks
    return node.state.board.side**2 - np.sum(visited)
    # return (np.sum(vertical) / 8 + np.sum(horizontal) / 4) * self.average



if __name__ == "__main__":
  # TODO:
  # Ler o ficheiro do standard input,
  # Usar uma técnica de procura para resolver a instância,
  # Retirar a solução a partir do nó resultante,
  # Imprimir para o standard output no formato indicado.
  sol = PipeMania(Board.parse_instance())
  # print(problem.initial.board.matrix)
  # print("Solução:")
  # sol = hill_climbing(problem).board
  # sys.setrecursionlimit(1500)
  sol = astar_search(sol).state.board.matrix
  # sol = sol.initial.board.matrix
  for row in sol[1:-1,1:-1]:
    print('\t'.join(map(lambda x: pieceToStr[x], row)))

