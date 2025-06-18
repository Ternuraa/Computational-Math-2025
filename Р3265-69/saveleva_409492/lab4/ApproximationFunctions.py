import math

from numpy import exp, log

from Approximation import Approximation
import numpy as np


class ApproximationFunctions(Approximation):

    def __calculateLinearFunctionCoefficient(self):
        SX = self.calculateSumX(1)
        SXX = self.calculateSumX(2)
        SY = self.calculateSumY()
        SXY = self.calculateSumXY(1)
        N = self.getN()
        delta = SXX * N - SX * SX
        delta1 = SXY * N - SX * SY
        delta2 = SXX * SY - SX * SXY
        a = delta1 / delta
        b = delta2 / delta
        self._setPX(self._calculateLinearValues(a, b)) 
        self._setEpsilon(self._calculateEpsilon())
        S = self._deviationMeasure()
        SD = self._standardDeviation()
        return {'SX': SX, 'SXX': SXX, 'SY': SY,
                'SXY': SXY, 'N': N, 'delta': delta,
                'delta1': delta1, 'delta2': delta2,
                'a': a, 'b': b, 'S': S, 'SD': SD}

    # Линейная функция
    def linearFunction(self):
        coefficients = self.__calculateLinearFunctionCoefficient()
        print('\t\t\tЛинейная аппроксимация')
        print(f"\tCистема уравнений:\n"
              f"\t\t{coefficients['SXX']}a + {coefficients['SX']}b = {coefficients['SXY']}\n"
              f"\t\t{coefficients['SX']}a + {coefficients['N']}b = {coefficients['SY']}")
        print(f"\tИз нее находим:\n"
              f"\t\tdelta = {coefficients['SXX']}*{coefficients['N']} - {coefficients['SX']}*{coefficients['SX']} = {coefficients['delta']}\n"
              f"\t\tdelta1 = {coefficients['SXY']}*{coefficients['N']} - {coefficients['SX']}*{coefficients['SY']} = {coefficients['delta1']}\n"
              f"\t\tdelta2 = {coefficients['SXX']}*{coefficients['SY']} - {coefficients['SX']}*{coefficients['SXY']} = {coefficients['delta2']}\n"
              f"\t\t\ta = {coefficients['delta1']}/{coefficients['delta']} = {coefficients['a']}\n"
              f"\t\t\tb = {coefficients['delta2']}/{coefficients['delta']} = {coefficients['b']}\n")
        function = f"P1(x) = {coefficients['a']}x + ({coefficients['b']})"
        self.__printCoefficients(coefficients, function)
        print(f"Коэффициент корреляции Пирсона: {self._coefficientCorrelation()}")
        return function, {'a': coefficients['a'], 'b': coefficients['b']}, coefficients['S'], coefficients['SD']

    def __calculatePolynomialSecondFunctionCoefficient(self): # Полиномиальная функция 2-й степени
        SX = self.calculateSumX(1)
        SXX = self.calculateSumX(2)
        SXXX = self.calculateSumX(3)
        SXXXX = self.calculateSumX(4)
        SY = self.calculateSumY()
        SXY = self.calculateSumXY(1)
        SXXY = self.calculateSumXY(2)
        N = self.getN()
        M1 = np.array([[N, SX, SXX], [SX, SXX, SXXX], [SXX, SXXX, SXXXX]])  # Матрица (левая часть системы)
        V1 = np.array([SY, SXY, SXXY])  # Вектор (правая часть системы)
        a0, a1, a2 = np.linalg.solve(M1, V1)
        self._setPX(self._calculatePolynomialSecondValues(a0, a1, a2))
        self._setEpsilon(self._calculateEpsilon())
        S = self._deviationMeasure()
        SD = self._standardDeviation()
        return {'SX': SX, 'SXX': SXX, 'SXXX': SXXX, 'SXXXX': SXXXX,
                'SY': SY, 'SXY': SXY, 'SXXY': SXXY, 'N': N,
                'a0': round(a0, 5), 'a1': round(a1, 5), 'a2': round(a2, 5), 'S': S, 'SD': SD}

    def polynomialSecondFunction(self):
        print('\t\t\tПолиномиальная функция 2-ой степени')
        coefficients = self.__calculatePolynomialSecondFunctionCoefficient()
        function = f"P2(x) = {coefficients['a0']} + ({coefficients['a1']}x)+ ({coefficients['a2']}x^2)"
        print(f"\tПолучаем систему уравнений:\n"
              f"\t\t{coefficients['N']}a0 + {coefficients['SX']}a1 + {coefficients['SXX']}a2 = {coefficients['SY']}\n"
              f"\t\t{coefficients['SX']}a0 + {coefficients['SXX']}a1 + {coefficients['SXXX']}a2= {coefficients['SXY']}\n"
              f"\t\t{coefficients['SXX']}a0 + {coefficients['SXXX']}a1 + {coefficients['SXXXX']}a2= {coefficients['SXXY']}\n")
        print(f"\tНайденные коэффициенты:\n"
              f"\t\ta0 = {coefficients['a0']}\n"
              f"\t\ta1 = {coefficients['a1']}\n"
              f"\t\ta2 = {coefficients['a2']}\n")
        self.__printCoefficients(coefficients, function)
        return function, {'a0': coefficients['a0'], 'a1': coefficients['a1'], 'a2': coefficients['a2']}, \
            coefficients['S'], coefficients['SD']

    def __calculatePolynomialThirdFunctionCoefficient(self): # Полиномиальная функция 3-й степени
        SX = self.calculateSumX(1)
        SXX = self.calculateSumX(2)
        SXXX = self.calculateSumX(3)
        SXXXX = self.calculateSumX(4)
        SXXXXX = self.calculateSumX(5)
        SXXXXXX = self.calculateSumX(6)
        SY = self.calculateSumY()
        SXY = self.calculateSumXY(1)
        SXXY = self.calculateSumXY(2)
        SXXXY = self.calculateSumXY(3)
        N = self.getN()
        M1 = np.array([[N, SX, SXX, SXXX], [SX, SXX, SXXX, SXXXX], [SXX, SXXX, SXXXX, SXXXXX],
                       [SXXX, SXXXX, SXXXXX, SXXXXXX]])  # Матрица (левая часть системы)
        V1 = np.array([SY, SXY, SXXY, SXXXY])  # Вектор (правая часть системы)
        a0, a1, a2, a3 = np.linalg.solve(M1, V1)
        self._setPX(self._calculatePolynomialThirdValues(a0, a1, a2, a3))
        self._setEpsilon(self._calculateEpsilon())
        S = self._deviationMeasure()
        SD = self._standardDeviation()
        return {'SX': SX, 'SXX': SXX, 'SXXX': SXXX, 'SXXXX': SXXXX, 'SXXXXX': SXXXXX, 'SXXXXXX': SXXXXXX,
                'SY': SY, 'SXY': SXY, 'SXXY': SXXY, 'SXXXY': SXXXY, 'N': N,
                'a0': round(a0, 5), 'a1': round(a1, 5), 'a2': round(a2, 5), 'a3': round(a3, 5), 'S': S, 'SD': SD}

    def polynomialThirdFunction(self):
        print('\t\t\tПолиномиальная функция 3-ей степени')
        coefficients = self.__calculatePolynomialThirdFunctionCoefficient()
        function = f"P3(x) = {coefficients['a0']} + ({coefficients['a1']}x)+ ({coefficients['a2']}x^2) + ({coefficients['a3']}x^3)"
        print(f"\tПолучим систему уравнений:\n"
              f"\t\t{coefficients['N']}a0 + {coefficients['SX']}a1 + {coefficients['SXX']}a2 + {coefficients['SXXX']}a3 = {coefficients['SY']}\n"
              f"\t\t{coefficients['SX']}a0 + {coefficients['SXX']}a1 + {coefficients['SXXX']}a2 + {coefficients['SXXXX']}a3= {coefficients['SXY']}\n"
              f"\t\t{coefficients['SXX']}a0 + {coefficients['SXXX']}a1 + {coefficients['SXXXX']}a2 + {coefficients['SXXXXX']}a3= {coefficients['SXXY']}\n"
              f"\t\t{coefficients['SXXX']}a0 + {coefficients['SXXXX']}a1 + {coefficients['SXXXXX']}a2 + {coefficients['SXXXXXX']}a3= {coefficients['SXXXY']}\n")
        print(f"\tНайденные коэффициенты:\n"
              f"\t\ta0 = {coefficients['a0']}\n"
              f"\t\ta1 = {coefficients['a1']}\n"
              f"\t\ta2 = {coefficients['a2']}\n"
              f"\t\ta3 = {coefficients['a3']}\n")
        self.__printCoefficients(coefficients, function)
        return function, {'a0': coefficients['a0'], 'a1': coefficients['a1'], 'a2': coefficients['a2'],
                          'a3': coefficients['a3']}, \
            coefficients['S'], coefficients['SD']

    # Экспоненциальная
    def __calculateExponentFunctionCoefficient(self):
        y_arr = self.getArrayY()
        self.setArrayY([log(y) for y in y_arr])
        A, B, a, b = self.calculateLinearForOtherFunctions()
        self.setArrayY(y_arr)
        self._setPX(self._calculateExponentValues(a, b))
        self._setEpsilon(self._calculateEpsilon())
        S = self._deviationMeasure()
        SD = self._standardDeviation()
        return {'A': A, 'B': B, 'a': a,
                'b': b, 'S': S, 'SD': SD}

    def exponentFunction(self):
        print('\t\t\tЭкспоненциальная функция')
        coefficients = self.__calculateExponentFunctionCoefficient()
        function = f"P4(x) = {coefficients['a']}*e({coefficients['b']}*x)"
        self.__printCoefficients(coefficients, function)
        return function, {'a': coefficients['a'], 'b': coefficients['b']}, \
            coefficients['S'], coefficients['SD']

    # Cтепенная
    def __calculateExponentialFunctionCoefficient(self):
        y_arr, x_arr = self.getArrayY(), self.getArrayX()
        self.setArrayY([log(y) for y in y_arr])
        self.setArrayX([log(x) for x in x_arr])
        A, B, a, b = self.calculateLinearForOtherFunctions()
        self.setArrayY(y_arr)
        self.setArrayX(x_arr)
        self._setPX(self._calculateExponentialValues(a, b))
        self._setEpsilon(self._calculateEpsilon())
        S = self._deviationMeasure()
        SD = self._standardDeviation()
        return {'A': A, 'B': B, 'a': a,
                'b': b, 'S': S, 'SD': SD}

    def exponentialFunction(self):
        print('\t\t\tСтепенная функция')
        coefficients = self.__calculateExponentialFunctionCoefficient()
        function = f"P5(x) = {coefficients['a']}*x^({coefficients['b']})"
        self.__printCoefficients(coefficients, function)
        return function, {'a': coefficients['a'], 'b': coefficients['b']}, \
            coefficients['S'], coefficients['SD']

    # Логарифмическая
    def __calculateLogarithmFunctionCoefficient(self):
        x_arr = self.getArrayX()
        self.setArrayX([log(x) for x in x_arr])
        coefficient = self.__calculateLinearFunctionCoefficient()
        A, B = coefficient['a'], coefficient['b']
        a, b = A, B
        self.setArrayX(x_arr)
        self._setPX(self._calculateLogarithmValues(a, b))
        self._setEpsilon(self._calculateEpsilon())
        S = self._deviationMeasure()
        SD = self._standardDeviation()
        return {'A': A, 'B': B, 'a': a,
                'b': b, 'S': S, 'SD': SD}

    def logarithmFunction(self):
        print('\t\t\tЛогарифмическая функция')
        coefficients = self.__calculateLogarithmFunctionCoefficient()
        function = f"P6(x) = {coefficients['a']}*ln(x)+{coefficients['b']}"
        self.__printCoefficients(coefficients, function)
        return function, {'a': coefficients['a'], 'b': coefficients['b']}, \
            coefficients['S'], coefficients['SD']

    def __printCoefficients(self, coefficients, function):
        for key, val in coefficients.items():
            print(f" {key} = {val}.")
        print(f"Аппроксимирующая функция имеет вид {function}")
        self._printTable()
        print(f"Мера отклонения S = {coefficients['S']}")
        print(f"Среднеквадратическое отклонение SD = {coefficients['SD']}")

    def calculateLinearForOtherFunctions(self):
        coefficient = self.__calculateLinearFunctionCoefficient()
        A, B = coefficient['b'], coefficient['a']
        a, b = math.exp(A), B
        return A, B, a, b

