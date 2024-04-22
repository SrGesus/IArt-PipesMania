# To-do List

## To-Do
- [x] Board.parse_instance()->Board
  - [x] Probably a Numpy ndarray [from stdin](https://stackoverflow.com/a/8192426) 
- [x] PipesMania.goal_test(state)->bool
- [ ] PipesMania.actions(state)->list
  - [ ] Don't allow connections into walls
  - [ ] Return list of possible actions
- [ ] Decide action type
(probably tuples if an action is just an index and rotation, look into [the benchmarks](benchmarking))
  - [ ] Consider if action should be turning a single piece or many
- [ ] PipesMania.result(state, action)->state
- [ ] PipesMania.h(state)->int,float


