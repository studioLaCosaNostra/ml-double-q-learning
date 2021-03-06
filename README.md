# ml-double-q-learning

Library implementing the double-q-learning algorithm.

paper: https://papers.nips.cc/paper/3964-double-q-learning.pdf

## Install

`npm install ml-double-q-learning`

## DoubleQLearningAgent

```typescript
export class DoubleQLearningAgent<TAction = any> implements IQLearningAgent {
  public replayMemory: [string, number, number][] = [];
  public episode: number = 0;
  public trained = false;

  constructor(
    public actions: TAction[],
    private pickActionStrategy: (actionsStats: number[], episode: number) => Promise<number> = greedyPickAction,
    public memory: IMemoryAdapter = new MapInMemory(),
    public learningRate = 0.1,
    public discountFactor = 0.99,
  ) {}

  public async play(state: IState): Promise<IStep<TAction>> {};

  public reward(step: IStep<TAction>, reward: number): void {};

  public async learn(): Promise<void> {};
}
```

## Memory (from ml-q-learning)

- [`MapInMemory`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/memory/map-in-memory.ts#L4)
- [`IndexedDBMemory`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/memory/indexeddb-memory.ts#L23)

## Pick action strategy (from ml-q-learning)

- [`randomPickAction`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/pick-action-strategy/index.ts#L13)
- [`greedyPickAction`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/pick-action-strategy/index.ts#L17)
- [`epsilonGreedyPickAction`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/pick-action-strategy/index.ts#L22)
- [`decayingEpsilonGreedyPickAction`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/pick-action-strategy/index.ts#L32)
- [`softmaxPickAction`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/pick-action-strategy/index.ts#L39)
- [`epsilonSoftmaxGreedyPickAction`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/pick-action-strategy/index.ts#L51)
- [`decayingEpsilonSoftmaxGreedyPickAction`](https://github.com/studioLaCosaNostra/ml-q-learning/blob/master/src/pick-action-strategy/index.ts#L61)

## Example use

`Maze escape`

[src/example/maze-escape.ts](https://github.com/studioLaCosaNostra/ml-double-q-learning/blob/master/src/example/maze-escape.ts)

```
P - Player
# - Wall
. - Nothing
X - Trap = -200
R - Treasure = 200
F - Finish = 1000
```

```bash
Start maze
[ [ 'P', '.', '.', '#', '.', '.', '.', '#', 'R' ],
  [ '.', '#', '.', '#', '.', '.', '.', '#', '.' ],
  [ '.', '#', '.', '#', '.', '#', '.', '#', '.' ],
  [ '.', '#', 'X', '#', '.', '#', '.', '.', '.' ],
  [ '.', '#', '#', '#', 'F', '#', '.', '.', '.' ],
  [ '.', '#', '.', '#', '#', '#', '.', '#', 'X' ],
  [ '.', '.', 'X', '.', '.', '.', '.', '#', '.' ],
  [ '.', '.', '.', '.', '#', '.', '.', '#', 'R' ] ]

...many plays...

-------------------------------
  numberOfPlay: 35702,
  score: 1168
  episode: 3322672
  memorySize: 968
-------------------------------

[ [ '.', '.', '.', '#', '.', '.', '.', '#', '.' ],
  [ '.', '#', '.', '#', '.', '.', '.', '#', '.' ],
  [ '.', '#', '.', '#', '.', '#', '.', '#', '.' ],
  [ '.', '#', 'X', '#', '.', '#', '.', '.', '.' ],
  [ '.', '#', '#', '#', 'P', '#', '.', '.', '.' ],
  [ '.', '#', '.', '#', '#', '#', '.', '#', 'X' ],
  [ '.', '.', 'X', '.', '.', '.', '.', '#', '.' ],
  [ '.', '.', '.', '.', '#', '.', '.', '#', 'R' ] ]
```

## Sources

- https://papers.nips.cc/paper/3964-double-q-learning.pdf
- https://towardsdatascience.com/double-q-learning-the-easy-way-a924c4085ec3