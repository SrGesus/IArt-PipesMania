# Benchmarking
These are just some benchmarks for making decisions on which data types to use.


Ran [[action_benchmark.py]] with N = 50000:
## Named Tuple
#### Named tuple with a, b, c:
0.0007412670001940569
#### Named tuple, using index:
0.0005207590002100915
#### Named tuple, using a local key:
0.0005243659998086514

## Tuple
#### Tuple with three values, using a constant key:
0.0005205350007599918
#### Tuple with three values, using a local key:
0.0005244349995336961

## Dictionary
#### Dictionary with keys a, b, c:
0.0005476189999171766
#### Dictionary with local keys a, b, c:
0.0005480349991557887
## Named Tuple
#### List with three values, using a constant key:
0.0005245729998932802
#### List with three values, using a local key:
0.0005261839996819617
## Numpy Array
#### Numpy Array with three values, using a constant key:       
0.0018109239999830606
#### Numpy Array with three values, using a local key:
0.001783569000508578