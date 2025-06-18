from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt


def show_2d(y: Callable, points: List[Tuple]):
    width = max(abs(points[0][0]), abs(points[len(points) - 1][0])) + 1
    height = abs(y(width))

    vf = np.vectorize(y)
    x = np.linspace(-width, width, 100)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.grid(True)
    plt.xlim((-width, width))
    plt.ylim((-height, height))

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.plot(x, vf(x), 'g', label='y=f(x)')
    ax.plot(*list(zip(*points)), 'ro')

    plt.show()
