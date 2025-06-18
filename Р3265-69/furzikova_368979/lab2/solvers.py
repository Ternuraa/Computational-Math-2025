from typing import Callable, Optional, List

import numpy as np
from attr import dataclass, field
from prettytable import PrettyTable


MAX_ITER_COUNT = 100


@dataclass
class Result:
    root: Optional[float] = None
    error: Optional[float] = None
    header: list = field(factory=list)
    data: list = field(factory=list)
    znach_f: float = field(default=0)
    iter_count: int = field(default=0)

    def __str__(self):
        tt = PrettyTable(self.header)
        data = [(n, *map(lambda x: f'{x:.3f}', floats)) for n, *floats in self.data]
        tt.add_rows(data)
        return str(tt)


@dataclass
class SystemResult:
    iteration: int = field(default=0)
    solved: bool = field(default=False)
    roots: list = field(factory=list)
    errors: list = field(factory=list)

    def __str__(self):
        if self.solved:
            return 'Решение: ' + ' '.join(f'{x:.3f}' for x in self.roots) + '\n' + \
            'Погрешности: ' + ' '.join(str(x) for x in self.errors) + '\n' + \
            f'Количество итераций {self.iteration}\n'
        else:
            return 'Решений не найдено!'


def horde_method(f: Callable, left: float, right: float, fix=-1, epsilon=10e-3):
    # [left, right] - интервал изоляции корня
    res = Result(header=['№', 'a', 'b', 'x', 'f(a)', 'f(b)', 'f(x)', '|a-b|'])

    x0 = left if fix == -1 else right
    for i in range(1, MAX_ITER_COUNT + 1):
        res.iter_count = i
        x1 = (left * f(right) - right * f(left)) / (f(right) - f(left))
        res.data.append([i, left, right, x1, f(left), f(right), f(x1), abs(left - right)])
        if abs(x1 - x0) <= epsilon or abs(f(x1)) <= epsilon:
            res.root = x1
            res.error = abs(x1 - x0)
            break
        if f(x1) * f(left) < 0:
            right = x1
        else:
            left = x1
        x0 = x1

    res.znach_f = f(res.root)
    return res


def newton_method(y: Callable, df: Callable, x0: float, epsilon=10e-3):
    res = Result(header=['№', 'x_k', 'f(x_k)', "f'(x_k)", 'x_{k+1}', '|x_k-x_{k+1}|'])

    for i in range(1, MAX_ITER_COUNT + 1):
        res.iter_count = i
        x1 = x0 - y(x0) / df(x0)
        res.data.append([i, x0, y(x0), df(x0), x1, abs(x1 - x0)])

        if abs(x1 - x0) <= epsilon or abs(y(x1) / df(x1)) <= epsilon or abs(y(x1)) <= epsilon:
            res.root = x1
            res.error = abs(x1 - x0)
            break
        x0 = x1
    res.znach_f = y(res.root)
    return res


def simple_iteration_method(f: Callable, phi: Callable, x0=1, epsilon=10e-3):
    res = Result(header=['№', 'x_k', 'f(x_k)', 'x_{k+1}', 'phi(x_k)', '|x_k-x_{k+1}|'])

    for i in range(1, MAX_ITER_COUNT + 1):
        res.iter_count = i
        x1 = phi(x0)

        res.data.append([i, x0, f(x0), x1, phi(x0), abs(x1 - x0)])

        if abs(x1 - x0) <= epsilon:
            res.root = x1
            res.error = abs(x1 - x0)
            break
        x0 = x1
    res.znach_f = f(res.root)
    return res


def _newton_method(f, jack, x_init):
    jacobian = jack(*x_init)
    vector_b_f_output = f(*x_init)
    x_delta = np.linalg.solve(jacobian, vector_b_f_output)
    x_plus_1 = x_delta + x_init
    return x_plus_1


def system_newton_method(f, jack, x_init, epsilon):
    result = SystemResult()

    x_old = x_init
    x_new = _newton_method(f, jack, x_old)
    diff = np.linalg.norm(x_old-x_new)

    while diff > epsilon or result.iteration == MAX_ITER_COUNT:
        x_new = _newton_method(f, jack, x_old)
        diff = np.linalg.norm(x_old-x_new)
        x_old = x_new
        result.iteration += 1
    convergent_val = x_new

    if result.iteration != MAX_ITER_COUNT:
        result.solved = True

    result.roots = convergent_val
    return result
