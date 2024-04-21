import numpy as np
from timeit import timeit


# arr = np.random.randint(1, 14, (50,50), dtype='uint8')
arr = np.zeros((10,10), dtype='uint8')

def test_np_bitops(m: np.ndarray):
  shifted = m << 1
  bottom = shifted[1:,] ^ m[:-1,] 
  bottom &= 0b1000
  if (np.any(bottom)):
    return False
  right = shifted[:,1:]
  right ^= m[:,:-1] 
  right &= 0b0010
  return not np.any(right)


def tests_lists(m: np.ndarray):
  m = m.tolist()
  for i in range(1, len(m)):
    for j in range(1, len(m)):
      if (m[i-1][j] ^ (m[i][j] << 1) & 0b1000):
        return False
      if (m[i-1][j] ^ (m[i][j] << 1) & 0b0010):
        return False
  return True
    
def tests_arr(m: np.ndarray):
  for i in range(1, len(m)):
    for j in range(1, len(m)):
      if (m[i-1][j] ^ (m[i][j] << 1) & 0b1000):
        return False
      if (m[i-1][j] ^ (m[i][j] << 1) & 0b0010):
        return False
  return True
      

if __name__ == '__main__':
  import timeit
  N = 10
  print("## Python List:")
  print(sorted(timeit.repeat("tests_lists(arr)", "from __main__ import arr, tests_lists", repeat=N, number=N))[N//2]*100/N)
  
  print("## Numpy Array with regular indexation:")
  print(sorted(timeit.repeat("tests_arr(arr)", "from __main__ import arr, tests_arr", repeat=N, number=N))[N//2]*100/N)

  print("## Numpy Array with matrices ops:")
  print(sorted(timeit.repeat("test_np_bitops(arr)", "from __main__ import arr, test_np_bitops", repeat=N, number=N))[N//2]*100/N)


