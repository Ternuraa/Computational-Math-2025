# variant number: 15


# 3 - Метод Ньютона
# 5 - Метод простой итерации
# 6 - Метод Ньютона


from math import sqrt

import numpy as np

from solvers import horde_method, newton_method, simple_iteration_method, system_newton_method
from graph import show_2d

A, B, C, D = -2.40, 1.27, 8.63, 2.31
f = lambda x: A * x ** 3 + B * x ** 2 + C * x + D
df = lambda x: 3 * A * x ** 2 + 2 * B * x + C
ddf = lambda x: 6 * A * x + 2 * B


def dphi(x):
    return abs(sqrt(abs(1/A)) / 3 * (B * x**2 + C * x + D) ** (-2/3) * (2*B*x + C))


def phi(x):
    base = -(B * x ** 2 + C * x + D) / A
    if base < 0:
        return -(-base) ** (1 / 3)
    return base ** (1 / 3)


def non_linear(output_file):
    left, right = map(float, input('Границы левого корня через пробел (-2 -1): ').split())
    #c_x0 = float(input('Нулевое приближение центрального корня (0): '))

    a, b = map(float, input('Границы для второго корня: (-1 0): ').split())
    if f(a) * ddf(a) > 0:
        c_x0 = a
    else:
        c_x0 = b

    print(f'x0 для второго корня равен равен: {c_x0}')

    # r_x0 = float(input('Нулевое приближение правого корня (5): '))
    a1, b1 = map(float, input('Границы для правого корня: (2 3): ').split())
    if f(a1) * ddf(a1) > 0:
        r_x0 = a1
    else:
        r_x0 = b1

    print(f'dpfi в a1 = {dphi(a1)}')
    print(f'dpfi в b1 = {dphi(b1)}')

    epsilon = float(input('Погрешность (0.01): '))

    left_x = horde_method(f, left=left, right=right, epsilon=epsilon)
    center_x = newton_method(f, df, c_x0, epsilon=epsilon)
    right_x = simple_iteration_method(f, phi, r_x0, epsilon=epsilon)

    show_2d(f, [(left_x.root, 0), (center_x.root, 0), (right_x.root, 0)])

    print(f'Левый корень x_0={left_x.root:.7f}, знач-e: {left_x.znach_f:.7f}, n: {left_x.iter_count}')
    # Значение функции + количество итераций
    print(f'Центральный корень x_1={center_x.root:.7f}, знач-e: {center_x.znach_f:.7f}, n: {center_x.iter_count}')
    # Значение функции + количество итераций
    print(f'Правый корень x_2={right_x.root:.7f}, знач-e: {right_x.znach_f:.7f}, n: {right_x.iter_count}')
    # Значение функции + количество итераций

    print(f'Таблица для метода хорд:')
    print(left_x)

    print('Таблица для метода Ньютана:')
    print(center_x)

    print('Таблица для метода простой итерации:')
    print(right_x)

    if output_file != None:
        with open(output_file, 'a') as fl:
            fl.write(f'Таблица для метода хорд:')
            fl.write(str(left_x))

            fl.write('Таблица для метода Ньютана:')
            fl.write(str(center_x))

            fl.write('Таблица для метода простой итерации:')
            fl.write(str(right_x))


def function_1(x, y):
    return [
        (-1)*(x+(2*y)-2),
        (-1)*((x**2)+(4*(y**2))-4)
    ]


def jacobian_1(x, y):
    return [
        [1, 2],
        [2*x, 8*y]
    ]


def function_2(x, y, z):
    return [
        (-1)*(x+y+z-3),
        (-1)*((x**2)+(y**2)+(z**2)-5),
        (-1)*((np.exp(x))+(x*y)-(x*z)-1)
    ]


def jacobian_2(x, y, z):
    return [[1, 1, 1], [2*x, 2*y, 2*z], [np.exp(x), x, -x]]


FUNCTIONS = [
    {
        'disp': 'f1(x1, x2) = x + 2*y - 2\nf2(x1, x2) = x^2 + 4*y^2 - 4)',
        'func': function_1,
        'jacob': jacobian_1,
        'init': [1, 1],
    },
    {
        'disp': 'f1(x1, x2, x3) = x + y + z - 3\nf2(x1, x2, x3) = x^2 + y^2 + z^2 - 5\nf3(x1, x2, x3) = exp(x) + x*y - x*z -1',
        'func': function_2,
        'jacob': jacobian_2,
        'init': [100, 200, 3],
    },
]


def non_linear_system(output_file):
    print(f'Выберите систему нелинейных уравнений:')
    for i, group in enumerate(FUNCTIONS, 1):
        print(f'Система №{i}')
        print(group['disp'])
        print()
    n = int(input())
    f = FUNCTIONS[n - 1]['func']
    hint = FUNCTIONS[n-1]['init']
    jacob = FUNCTIONS[n - 1]['jacob']

    x0 = list(map(float, input(f'Начальные приближения ({" ".join(str(num) for num in hint)}): ').split()))
    eps = float(input('Погрешность (0.001): '))

    res = system_newton_method(f, jacob, x0, eps)

    print(res)
    if output_file != None:
        with open(output_file, 'a') as fl:
            fl.write(f'Решение системы нелинейных уравнений:')
            fl.write(str(res))

def main():
    read_from_file = input('Нужно ли записать результат в файл (y/n): ')
    output_file = None
    if(read_from_file == 'y'):
        output_file = input('Введите название файла (out.txt): ')


    print('Решение нелейного уравнения: ')
    non_linear(output_file)
    print('Решение системы нелинейных уравнений: ')
    non_linear_system(output_file)


if __name__ == '__main__':
    main()
