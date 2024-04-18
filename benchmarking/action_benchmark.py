from collections import namedtuple
import numpy as np

STest = namedtuple("TEST", "a b c")
a = STest(a=1,b=2,c=3)

c = {'a':1, 'b':2, 'c':3}
akey = 'a'

d = (1,2,3)
e = [1,2,3]
key = 2

f = np.array([1,2,3])

N=100000
if __name__ == '__main__':
  import timeit

  print("## Named Tuple")
  print("#### Named tuple with a, b, c:")
  print(sorted(timeit.repeat("z = a.c", "from __main__ import a", repeat=N, number=N))[N//2])

  print("#### Named tuple, using index:")
  print(sorted(timeit.repeat("z = a[2]", "from __main__ import a", repeat=N, number=N))[N//2])
  
  print("#### Named tuple, using a local key:")
  print(sorted(timeit.repeat("z = a[key]", "from __main__ import a, key", repeat=N, number=N))[N//2])

  print("\n## Tuple")
  print("#### Tuple with three values, using a constant key:")    
  print(sorted(timeit.repeat("z = d[2]", "from __main__ import d", repeat=N, number=N))[N//2])

  print("#### Tuple with three values, using a local key:")
  print(sorted(timeit.repeat("z = d[key]", "from __main__ import d, key", repeat=N, number=N))[N//2])

  print("\n## Dictionary")
  print("#### Dictionary with keys a, b, c:")
  print(sorted(timeit.repeat("z = c['c']", "from __main__ import c", repeat=N, number=N))[N//2])
  
  print("#### Dictionary with local keys a, b, c:")
  print(sorted(timeit.repeat("z = c[akey]", "from __main__ import c, akey", repeat=N, number=N))[N//2])

  print("## Named Tuple")
  print("#### List with three values, using a constant key:")
  print(sorted(timeit.repeat("z = e[2]", "from __main__ import e", repeat=N, number=N))[N//2])

  print("#### List with three values, using a local key:")
  print(sorted(timeit.repeat("z = e[key]", "from __main__ import e, key", repeat=N, number=N))[N//2])

  print("## Numpy Array")
  print("#### Numpy Array with three values, using a constant key:")
  print(sorted(timeit.repeat("z = f[2]", "from __main__ import f", repeat=N, number=N))[N//2])
  
  print("#### Numpy Array with three values, using a local key:")
  print(sorted(timeit.repeat("z = f[key]", "from __main__ import f, key", repeat=N, number=N))[N//2])