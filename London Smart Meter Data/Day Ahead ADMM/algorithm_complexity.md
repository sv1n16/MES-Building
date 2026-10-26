# Algorithm Complexity

Let:

- `N` = number of buildings
- `T` = number of time periods
- `K` = number of ADMM iterations
- `S_T` = cost of solving one exchange-ADMM local subproblem
- `S_NT` = cost of solving one bilateral-ADMM local subproblem

| Method | Time complexity | Memory complexity |
|---|---:|---:|
| Rule-based sharing | `O(T N^2)` | `O(T N^2)` for trade results |
| Central optimisation | Model size `O(N^2 T)`; MIQCP solve is NP-hard in the worst case | `O(N^2 T)` |
| Exchange ADMM | `O(K N S_T + K N T)` | `O(N T)` |
| Bilateral ADMM | `O(K N S_NT + K N^2 T)` | `O(N^2 T)` |

## Interpretation

### Rule-based sharing

Each time period may compare every building with every other building when matching donors and receivers:

```text
O(T N^2)
```

It is the most scalable method because it does not call an optimisation solver.

### Central optimisation

The model contains building-to-building sharing variables for every ordered pair and time period:

```text
N^2 T
```

The formulation also contains binary battery states and quadratic/bilinear terms. Therefore, worst-case solution time is exponential and depends strongly on the solver, time limit, and relaxation settings.

### Exchange ADMM

Each iteration solves `N` independent local problems, each with approximately `O(T)` variables, followed by an `O(N T)` consensus update:

```text
O(K N S_T + K N T)
```

Its stored state scales approximately linearly with the number of buildings:

```text
O(N T)
```

### Bilateral ADMM

There are approximately `N(N-1)/2` trading pairs. Each building stores `N-1` local trade copies, giving approximately `O(N^2 T)` total trade variables. Pairwise consensus updates also require:

```text
O(K N^2 T)
```

Its memory requirement is:

```text
O(N^2 T)
```

Bilateral ADMM therefore scales much worse than exchange ADMM as the community grows.
