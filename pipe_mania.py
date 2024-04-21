from sys import stdin
from typing_extensions import Self, Union
import numpy as np
from collections import defaultdict

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

pieceToStr: list = ['', 'FD', 'FE', 'LH', 'FB', 'VB', 'BB', 'VE', 'FC', 'VD', 'VC', 'BC', 'LV', 'BD', 'BE']
pieceToAction: list = ['', 'FD', 'FE', 'LH', 'FB', 'VB', 'BB', 'VE', 'FC', 'VD', 'VC', 'BC', 'LV', 'BD', 'BE']

class Board:
  """ Internal representation of a PipeMania grid."""
  def __init__(self, matrix: np.ndarray) -> None:
     self.matrix: np.ndarray = matrix

  @staticmethod
  def parse_instance() -> Self:
      """Reads a board from stdin"""
      output = np.pad(np.genfromtxt(stdin, dtype='S2'), 1)
      matrix = np.ndarray(output.shape, dtype='uint8')
      for k in strToPiece:
         matrix[output == k] = strToPiece[k]
      return Board(matrix)

  def adjacent_values(self, row: int, col: int) -> tuple[np.uint8, ...]:
    """ Returns the values of the adjacent cells on top, 
    bottom, left, and right of the given position"""
    
    return (self.matrix[row-1, col], self.matrix[row+1, col],
            self.matrix[row, col-1], self.matrix[row+1, col])

def goal_tests(m: np.ndarray):
  shifted = m << 1
  bottom = shifted[1:,].copy()
  right = shifted[:,1:]
  bottom ^= m[:-1,]
  right ^= m[:,:-1]
  bottom &= 0b1000
  right &= 0b0010
  return not np.any(bottom) and not np.any(right)

  
