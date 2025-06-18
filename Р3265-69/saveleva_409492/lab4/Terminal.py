from numpy import exp, log

import numpy as np

from ApproximationFunctions import ApproximationFunctions
from Validation.Validator import Validator
from Exceptions.IncorrectValueException import IncorrectValueException
import matplotlib.pyplot as plt

INTERVAL_FOR_GRAPHIC_X = 1


class Terminal:
    def __init__(self):
        self.__isFromFile: bool = False

    def isFile(self, mode) -> bool:
        if mode == 0:
            print("Выберите формат ввода(c клавиатуры - k или из файла - f):")
        else:
            print("Хотите записать результаты в файл?(нет - k или  да - f):")
        is_file = input()
        if is_file.__eq__('k'):
            return False
        elif is_file.__eq__('f'):
            return True
        else:
            return self.isFile(mode)

    def refresh(self):
        self.__isFromFile = False

    def work(self):
        try:
            print('\t\t\tАппроксимация функции методом наименьших квадратов')
            self.__isFromFile = self.isFile(0)
            if self.__isFromFile:
                x_array, y_array = self.enterArraysFromFile()
            else:
                x_array, y_array = self.enterArraysFromKeyboard()
            min_x, max_x = min(x_array), max(x_array)
            x = np.arange(min_x, max_x, 0.1)
            self.__isFromFile = self.isFile(1)
            results = list()
            approximation = ApproximationFunctions(x_array, y_array)
            results.append(approximation.linearFunction())
            results.append(approximation.polynomialSecondFunction())
            results.append(approximation.polynomialThirdFunction())
            results.append(approximation.exponentialFunction())
            results.append(approximation.exponentFunction())
            results.append(approximation.logarithmFunction())
            self.printTableOfCompare(results)
            index = self.minSD(results)
            print(f'Наилучшее приближение :\n'
                  f'1.Аппроксимирующая функция: {results[index][0]}\n'
                  f'2. Коэффициенты:{results[index][1]}\n'
                  f'3. Среднеквадратичное отклонение: {results[index][3]}')

            plt.scatter(x_array, y_array, color='red', s=40, marker='o', label='f(x)')
            plt.plot(x, [results[0][1]['a'] * x + results[0][1]['b'] for x in x], color='green', label=results[0][0])
            plt.plot(x,
                     [results[1][1]['a0'] + x * results[1][1]['a1'] + results[1][1]['a2'] * x ** 2 for x in x],
                     '--',
                     color='blue', label=results[1][0])
            plt.plot(x,
                     [results[2][1]['a0'] + x * results[2][1]['a1'] + results[2][1]['a2'] * x ** 2 + results[2][1][
                         'a3'] * x ** 3 for x in x], '--',
                     color='black', label=results[2][0])
            plt.plot(x,
                     [results[5][1]['a'] * log(x) + results[5][1]['b'] for x in x], '--',
                     color='purple', label=results[5][0])
            plt.legend()
            plt.grid()
            plt.figure(2)
            plt.plot(x, [results[3][1]['a'] * x ** results[3][1]['b'] for x in x], '--',
                     color='pink', label=results[3][0])
            plt.scatter(x_array, y_array, color='red', s=40, marker='o', label='f(x)')
            plt.legend()
            plt.grid()
            plt.figure(3)
            plt.plot(x,
                     [results[4][1]['a'] * exp(x * results[4][1]['b']) for x in x], '--',
                     color='orange', label=results[4][0])
            plt.scatter(x_array, y_array, color='red', s=40, marker='o', label='f(x)')
            plt.legend()
            plt.grid()
        except IncorrectValueException as e:
            print(e.message)
            return
        except ValueError as e:
            print(e.message)
            return
        except RuntimeWarning:
            pass
        finally:
            plt.show()

    def enterArray(self, variable) -> object:
        try:
            print(f'Введите значения {variable} (через ;):')
            arr = [Validator.validateNumber(x) for x in input().split(';')]
            Validator.validateArraySize(arr)
            return arr
        except IncorrectValueException as e:
            print(e.message)
            return self.enterArray(variable)

    def enterArraysFromKeyboard(self):
        x_array = self.enterArray('x')
        y_array = self.enterArray('y')
        Validator.validateEqualityLengths(x_array, y_array)
        return x_array, y_array

    def enterArraysFromFile(self):
        try:
            f = open(input('Введите путь к файлу:'), 'r')
            x_array = [Validator.validateNumber(x) for x in f.readline().split(';')]
            Validator.validateArraySize(x_array)
            y_array = [Validator.validateNumber(x) for x in f.readline().split(';')]
            Validator.validateArraySize(y_array)
            Validator.validateEqualityLengths(x_array, y_array)
            return x_array, y_array
        except IncorrectValueException as e:
            print(e.message)
            return self.enterArraysFromFile()
        except FileNotFoundError as e:
            print(e.strerror)
            return self.enterArraysFromFile()

    def printTableOfCompare(self, results):
        print('\t\t\tРезультаты ')
        print('| Вид функции        |      Коэффициенты        '
              '|   Мера отклонения    |  Среднеквадратичное отклонение    |')
        for i in range(len(results)):
            print(f'| {results[i][0]} | ', end='')
            for key, val in results[i][1].items():
                print(f'{key} = {val}|', end='')
            print(f"| {results[i][2]} | {results[i][3]} |")

    def minSD(self, results):
        min = results[0][3]
        index = 0
        for i in range(len(results)):
            if results[i][3] < min:
                index = i
        return index