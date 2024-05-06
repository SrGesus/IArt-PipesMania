# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 23:
# 107030 Gabriel dos Reis Fonseca Castelo Ferreira
# 106537 Francisco Rodrigues Martins Gonçalves Fernandes

import sys
import numpy as np
from search import (
  Problem,
  Node,
  astar_search,
  breadth_first_tree_search,
  depth_first_tree_search,
  greedy_search,
  recursive_best_first_search,
)

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
    self.side = matrix.shape[0] - 1 # range() é exclusivo no ultimo elemento

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
    self.moves = [[pieceToAction[x]+[x] for x in row] for row in board.matrix]
    for i in range(1, board.side):
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
          self.moves[i][j] = [action for action in self.moves[i][j] if action != 0b0001]
          self.moves[i][j+1] = [action for action in self.moves[i][j+1] if action != 0b0010]
        if 0b0100 in self.moves[i][j] and 0b1000 in self.moves[i+1][j]: # up and down
          self.moves[i][j] = [action for action in self.moves[i][j] if action != 0b0100]
          self.moves[i+1][j] = [action for action in self.moves[i+1][j] if action != 0b1000]


  def actions(self, state: PipeManiaState):
    """Retorna uma lista ou iterador de ações que podem ser executadas a
    partir do estado passado como argumento."""
    row = state.depth // (state.board.side-1) + 1
    col = state.depth % (state.board.side-1) + 1
    if (row > state.board.side-1 or col > state.board.side-1):
      return
    for action in self.moves[row][col]:
      yield (row, col, action)

    # for row in range(1, state.board.side):
    #   for col in range(1, state.board.side):
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
    m = state.board.matrix
    shifted = m << 1
    vertical = shifted[:-1,] ^ m[1:,]
    vertical &= 0b1000
    if (np.any(vertical)):
      return False
    horizontal = shifted[:,:-1] ^ m[:,1:]
    horizontal &= 0b0010
    return not np.any(horizontal)

  def h(self, node: Node):
    """Função heuristica utilizada para a procura A*."""
    m = node.state.board.matrix
    shifted = m << 1
    vertical = shifted[:-1,] ^ m[1:,]
    vertical &= 0b1000
    horizontal = shifted[:,:-1] ^ m[:,1:]
    horizontal &= 0b0010
    return (np.sum(vertical) / 8 + np.sum(horizontal) / 4) * 1.5

  # TODO: outros metodos da classe


if __name__ == "__main__":
  # TODO:
  # Ler o ficheiro do standard input,
  # Usar uma técnica de procura para resolver a instância,
  # Retirar a solução a partir do nó resultante,
  # Imprimir para o standard output no formato indicado.
  problem = PipeMania(Board.parse_instance())
  # print(problem.initial.board.matrix)
  # print("Solução:")
  sol = recursive_best_first_search(problem).state.board.matrix
  for row in sol[1:-1,1:-1]:
    print('\t'.join(map(lambda x: pieceToStr[x], row)))

