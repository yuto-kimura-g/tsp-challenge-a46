import sys
import time
import dataclasses
import math
import random
import typing


StateType = typing.TypeVar("StateType", bound="State")
SolverType = typing.TypeVar("SolverType", bound="Solver")


def eprint(*args):
    print(*args, file=sys.stderr)


class Timer:
    # TL 1sec
    # 750[msec](LOCAL) --> 950[msec](JUDGE)
    TIME_LIMIT = 750  # [msec]
    # TIME_LIMIT = 5000  # [msec]

    def __init__(self) -> None:
        self.start: float = time.perf_counter()
        return

    def elapsed(self) -> float:
        """ return elapsed time [msec]  """
        return (time.perf_counter() - self.start) * 1000


@dataclasses.dataclass
class Node:
    x: int
    y: int


class State:
    def __init__(self, ans: list[int] = list(), obj: int = 0) -> None:
        # ans[i] := i番目に訪れる地点
        # ans[0]=ans[-1]=1
        self.ans: list[int] = ans[:]
        # obj := 移動距離の総和
        self.obj: int = obj
        return

    def copy(self, state: StateType) -> None:
        self.ans = state.ans[:]
        self.obj = state.obj
        return

    def update(self, i: int, j: int, delta: int) -> None:
        # 2-opt
        # cur route: ... - i - i+1 - ... - j-1 - j - j+1 - ...
        # new route: ... - i - j - j-1 - ... - i+1 - j+1 - ...
        # reverse sub-path [i+1, ..., j]
        self.ans[i+1:j+1] = list(reversed(self.ans[i+1:j+1]))
        # update obj
        self.obj += delta
        return

    def output(self, solver: SolverType) -> None:
        assert len(self.ans) == solver.N + 1, \
            (self.ans, len(self.ans), solver.N + 1)
        assert self.ans[0] == 1, self.ans
        assert self.ans[-1] == 1, self.ans
        assert set(self.ans[:-1]) == set(range(1, solver.N + 1)), self.ans
        print(*self.ans, sep="\n")
        return


class Solver:
    def __init__(self) -> None:
        self.timer = Timer()
        # param
        self.N = int(input())
        self.city: list[Node] = list()
        self.city.append(Node(-1, -1))  # 1-indexed
        for _ in range(self.N):
            x, y = map(int, input().split())
            self.city.append(Node(x=x, y=y))
        return

    def solve(self) -> State:
        best_state = State()
        best_state.ans = list(range(1, self.N + 1)) + [1]
        best_state.obj = self.eval_obj(best_state)
        eprint("init obj:", best_state.obj)
        best_state = self.local_search(best_state)
        return best_state

    def local_search(self, best_state: State) -> None:

        def eval_diff(state: State, i: int, j: int) -> float:
            # 2-opt
            # cur route: ... - i - i+1 - ... - j-1 - j - j+1 - ...
            # new route: ... - i - j - j-1 - ... - i+1 - j+1 - ...
            u, uu = state.ans[i], state.ans[(i + 1) % len(state.ans)]
            v, vv = state.ans[j], state.ans[(j + 1) % len(state.ans)]
            cur_dist = self.dist(u, uu) + self.dist(v, vv)
            new_dist = self.dist(u, v) + self.dist(uu, vv)
            return new_dist - cur_dist

        iter_limit = pow(10, 9)
        # SA param
        start_temp = 1000  # max(delta)
        end_temp = 1  # min(delta)
        # log
        # history = list()
        for iter in range(iter_limit):
            if self.timer.elapsed() >= self.timer.TIME_LIMIT:
                eprint("Time Out")
                break
            i = random.randrange(1, self.N - 1)
            j = random.randrange(i + 1, self.N)
            delta = eval_diff(best_state, i, j)
            # temp = T(t) := pow(start_temp,(1-t)) * pow(end_temp,t)
            # t=0でtemp=start_temp, t=1でtemp=end_tempとなる，幾何冷却スケジューリング
            t = self.timer.elapsed() / self.timer.TIME_LIMIT
            temp = pow(start_temp, 1 - t) * pow(end_temp, t)
            if delta < 0:
                # 改善するなら必ず採用
                best_state.update(i, j, delta)
                eprint("obj:", best_state.obj, i, j, iter)
            elif math.exp(-delta / temp) > random.random():
                # 改悪する場合も，遷移確率 P(d,T)=exp(-d/T) に従って採用
                # delta が average(start_temp, end_temp) = 500 くらいを考える
                # temp=start_tempの時，exp(-delta/temp) = exp(-500/1000) = 0.6 くらい
                #   -> 3回に2回くらい改悪を採用
                # temp=end_tempの時，exp(-delta/temp) = exp(-500/1) = 1e-200 = 0
                #   -> 改悪しない
                best_state.update(i, j, delta)
                eprint("obj:", best_state.obj, i, j, iter)
            # history.append((
            #     iter,
            #     best_state.obj
            #     # self.timer.elapsed(),
            #     # self.eval_score(best_state)
            # ))

        # import matplotlib.pyplot as plt
        # x, y = zip(*history)
        # plt.plot(x, y)
        # plt.show()
        return best_state

    def dist(self, i: int, j: int) -> float:
        """ city[i] <--dist--> city[j]  """
        ci = self.city[i]
        cj = self.city[j]
        return math.sqrt(pow(ci.x - cj.x, 2) + pow(ci.y - cj.y, 2))

    def eval_obj(self, state: State) -> float:
        d = 0.0
        for i in range(len(state.ans) - 1):
            d += self.dist(state.ans[i], state.ans[i + 1])
        return d

    def eval_score(self, state: State) -> float:
        return math.floor(pow(10, 6) / self.eval_obj(state))


if __name__ == '__main__':
    random.seed(0)
    solver = Solver()
    best_state = solver.solve()
    best_state.output(solver)
    eprint(f"score:{solver.eval_score(best_state):.3f}")
    eprint(f"elapsed:{solver.timer.elapsed():.3f}[msec]")
