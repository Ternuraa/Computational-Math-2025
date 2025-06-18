from Functions import calculateFunction, FUNCTIONS
from NumericalIntegrations.NumericalIntegration import NumericalIntegration


class MethodSympson(NumericalIntegration):

    def iterateSympsonlMethod(self, number_of_function):
        eps = self.getEpsilon()
        iterations = 0
        while True:
            print(f'Номер итерации №{iterations}')
            square_1, n_1 = self.sympsonMethod(number_of_function=number_of_function, n=self.getN())
            self.doubleN()
            square_2, n_2 = self.sympsonMethod(number_of_function=number_of_function, n=self.getN())
            if self.ruleRunge(square_1, square_2, 2, self.getEpsilon()):
                print(f"Решение интеграла {FUNCTIONS[number_of_function]['FUNCTION']} "
                      f"с границами a={self.getLeftBorder()} и b={self.getRightBorder()}:\n\t\t\t "
                      f"{square_2} \nдля разбиения на {n_2} отрезков.")
                break
            iterations += 1

    def sympsonMethod(self, number_of_function, n):
        a = self.getLeftBorder()
        b = self.getRightBorder()
        h = (b - a) / n
        iterations = [[0.0 for x in range(self.AMOUNT_OF_COLUMNS)] for i in range(n + 1)]
        square_even = 0
        square_odd = 0
        for i in range(n + 1):
            iterations[i][0] = i
            iterations[i][1] = a if i == 0 else a + h
            iterations[i][2] = calculateFunction(number_of_function, iterations[i][1])
            if i != 0 and i != n:
                if i % 2 == 0:
                    square_even += iterations[i][2]
                else:
                    square_odd += iterations[i][2]
            a = iterations[i][1]
        square = h / 3 * (iterations[0][2] + iterations[-1][2] + 2 * square_even + 4 * square_odd)
        print(f'\nМетод Симпсона. Численное интегрирование для n={n} и h={h}')
        self.printTableForMethods(iterations, method='trapezoidal')
        print(f'Найденный ответ:{square} для {n} отрезков разбиения.')
        return square, n