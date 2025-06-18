from Functions import calculateFunction, FUNCTIONS
from NumericalIntegrations.NumericalIntegration import NumericalIntegration


class MethodTrapezoidal(NumericalIntegration):

    def iterateTrapezoidalMethod(self, number_of_function):
        eps = self.getEpsilon()
        iterations = 0
        while True:
            print(f'Номер итерации №{iterations}')
            square_1, n_1 = self.trapezoidalMethod(number_of_function=number_of_function, n=self.getN())
            self.doubleN()
            square_2, n_2 = self.trapezoidalMethod(number_of_function=number_of_function, n=self.getN())
            if self.ruleRunge(square_1, square_2, 2, self.getEpsilon()):
                print(f"Решение интеграла {FUNCTIONS[number_of_function]['FUNCTION']} "
                      f"с границами a={self.getLeftBorder()} и b={self.getRightBorder()}:\n\t\t\t "
                      f"{square_2} \nдля разбиения на {n_2} отрезков.")
                break
            iterations += 1

    def trapezoidalMethod(self, number_of_function, n):
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
            if i != 0 and i != n:
                square += iterations[i][2]
            a = iterations[i][1]
        square = h * ((iterations[0][2] + iterations[-1][2]) / 2 + square)
        print(f'\nМетод трапеций. Численное интегрирование для n={n} и h={h}')
        self.printTableForMethods(iterations, method='trapezoidal')
        print(f'Найденный ответ:{square} для {n} отрезков разбиения.')
        return square, n
