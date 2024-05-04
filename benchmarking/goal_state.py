import numpy as np
from timeit import timeit


# arr = np.random.randint(1, 14, (50,50), dtype='uint8')
arr = np.zeros((50,50), dtype='uint8')
# arr[1,48] = 0b0100
# arr[2,48] = 0b1010
# arr[2,47] = 0b0000

arr2 = np.zeros((50,200), dtype=bool)
# arr2[1, 48*4+1] = True
# arr2[2, 48*4+0] = True

def test_np_bitops(m: np.ndarray):
  shifted = m << 1
  vertical = shifted[:-1,] ^ m[1:,]
  vertical &= 0b1000
  if (np.any(vertical)):
    return False
  horizontal = shifted[:,:-1] ^ m[:,1:]
  horizontal &= 0b0010
  return not np.any(horizontal)

def tests_np_separate_booleans(m: np.ndarray):
  return not np.any(m[1:,::4] ^ m[:-1,1::4]) and not np.any(m[:,3:-4:4] ^ m[:,6::4])

# def tests_lists(m: np.ndarray):
#   m = m.tolist()
#   for i in range(1, len(m)):
#     for j in range(1, len(m)):
#       if ((m[i-1][j] << 1 ^ m[i][j]) & 0b1000):
#         return False
#       if ((m[i][j-1] << 1 ^ m[i][j]) & 0b0010):
#         return False
#   return True
    
def tests_arr(m: np.ndarray):
  for i in range(1, len(m)):
    for j in range(1, len(m)):
      if ((m[i-1][j] << 1 ^ m[i][j]) & 0b1000):
        return False
      if ((m[i][j-1] << 1 ^ m[i][j]) & 0b0010):
        return False
  return True
      

if __name__ == '__main__':
  import timeit
  N = 10
  print("## Python List:")
  print(sorted(timeit.repeat("tests_lists(arr2)", "from __main__ import arr2, tests_lists", repeat=N, number=N))[N//2]*100/N)
  print(tests_lists(arr2))

  print("## Numpy Array with regular indexation:")
  print(sorted(timeit.repeat("tests_arr(arr)", "from __main__ import arr, tests_arr", repeat=N, number=N))[N//2]*100/N)
  print(tests_arr(arr))

  print("## Numpy Array with matrices ops:")
  print(sorted(timeit.repeat("test_np_bitops(arr)", "from __main__ import arr, test_np_bitops", repeat=N, number=N))[N//2]*100/N)
  print(test_np_bitops(arr))

