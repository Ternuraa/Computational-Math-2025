from Functions import *
from NumericalIntegrations.NumericalIntegration import *


K = 2


class MethodRectangles(NumericalIntegration):

    def iterateRightRectangles(self, number_of_function):
        eps = self.getEpsilon()
        iterations = 0
        while True:
            print(f'Номер итерации №{iterations}')
            square_1, n_1 = self.rightRectangles(number_of_function=number_of_function, n=self.getN())
            self.doubleN()
            square_2, n_2 = self.rightRectangles(number_of_function=number_of_function, n=self.getN())
            if self.ruleRunge(square_1, square_2, 2, self.getEpsilon()):
                print(f"Решение интеграла {FUNCTIONS[number_of_function]['FUNCTION']} "
                      f"с границами a={self.getLeftBorder()} и b={self.getRightBorder()}:\n\t\t\t "
                      f"{square_2} \nдля разбиения на {n_2} отрезков.")
                break
            iterations += 1

    def rightRectangles(self, number_of_function, n):
        a = self.getLeftBorder()
        b = self.getRightBorder()
        h = (b - a) / n
        iterations = [[0.0 for x in range(self.AMOUNT_OF_COLUMNS)] for i in range(n + 1)]
        square = 0
        for i in range(n + 1):
            iterations[i][0] = i
            if i == 0:
                iterations[i][1] = a
            else:
                iterations[i][1] = a + h
            iterations[i][2] = calculateFunction(number_of_function, iterations[i][1])
            if i != 0:
                square += iterations[i][2]
            a = iterations[i][1]
        square *= h
        print(f'\nМетод правых прямоугольников. Численное интегрирование для n={n} и h={h}')
        self.printTableForMethods(iterations, method='right')
        print(f'Найденный ответ:{square} для {n} отрезков разбиения.')
        return square, n

    def iterateLeftRectangles(self, number_of_function):
        eps = self.getEpsilon()
        iterations = 0
        while True:
            print(f'Номер итерации №{iterations}')
            square_1, n_1 = self.leftRectangles(number_of_function=number_of_function, n=self.getN())
            self.doubleN()
            square_2, n_2 = self.leftRectangles(number_of_function=number_of_function, n=self.getN())
            if self.ruleRunge(square_1, square_2, 2, self.getEpsilon()):
                print(f"Решение интеграла {FUNCTIONS[number_of_function]['FUNCTION']} "
                      f"с границами a={self.getLeftBorder()} и b={self.getRightBorder()}:\n\t\t\t "
                      f"{square_2} \nдля разбиения на {n_2} отрезков.")
                break
            iterations += 1

    def leftRectangles(self, number_of_function, n):
        a = self.getLeftBorder()
        b = self.getRightBorder()
        h = (b - a) / n
        iterations = [[0.0 for x in range(self.AMOUNT_OF_COLUMNS)] for i in range(n + 1)]
        square = 0
        for i in range(1, n + 2):
            if i == 1:
                iterations[i - 1][1] = a
            else:
                iterations[i - 1][1] = a + h
            iterations[i - 1][0] = i - 1
            iterations[i - 1][2] = calculateFunction(number_of_function, iterations[i - 1][1])
            if i != n + 1:
                square += iterations[i - 1][2]
            a = iterations[i - 1][1]
        square *= h
        print(f'\nМетод левых прямоугольников. Численное интегрирование для n={n} и h={h}')
        self.printTableForMethods(iterations, method='left')
        print(f'Найденный ответ:{square} для {n} отрезков разбиения.')
        return square, n

    def iterateMiddleRectangles(self, number_of_function):
        eps = self.getEpsilon()
        iterations = 0
        while True:
            print(f'Номер итерации №{iterations}')
            square_1, n_1 = self.middleRectangles(number_of_function=number_of_function, n=self.getN())
            self.doubleN()
            square_2, n_2 = self.middleRectangles(number_of_function=number_of_function, n=self.getN())
            if self.ruleRunge(square_1, square_2, 2, self.getEpsilon()):
                print(f"Решение интеграла {FUNCTIONS[number_of_function]['FUNCTION']} "
                      f"с границами a={self.getLeftBorder()} и b={self.getRightBorder()}:\n\t\t\t "
                      f"{square_2} \nдля разбиения на {n_2} отрезков.")
                break
            iterations += 1

    def middleRectangles(self, number_of_function, n):
        a = self.getLeftBorder()
        b = self.getRightBorder()
        h = (b - a) / n
        iterations = [[0.0 for x in range(self.AMOUNT_OF_COLUMNS)] for i in range(n + 1)]
        square = 0
        for i in range(1, n + 1):
            iterations[i][0] = i
            if i == 1:
                iterations[i][1] = (a + a + h) / 2
            else:
                iterations[i][1] = a + h
            iterations[i][2] = calculateFunction(number_of_function, iterations[i][1])
            square += iterations[i][2]
            a = iterations[i][1]
        square *= h
        print(f'\nМетод средних прямоугольников. Численное интегрирование для n={n} и h={h}')
        self.printTableForMethods(iterations, method='middle')
        print(f'Найденный ответ:{square} для {n} отрезков разбиения.')
        return square, n