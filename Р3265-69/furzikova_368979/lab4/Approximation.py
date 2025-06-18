from numpy import sqrt, log, exp


class Approximation:

    def __init__(self, x_arr, y_arr):
        self.__x_array = x_arr
        self.__y_array = y_arr
        self.__n = len(self.__x_array)
        self.__p_x = []
        self.__epsilon = []

    def getArrayX(self):
        return self.__x_array

    def setArrayX(self, arr):
        self.__x_array = arr

    def getArrayY(self):
        return self.__y_array

    def setArrayY(self, arr):
        self.__y_array = arr

    def getN(self):
        return self.__n

    def getEpsilon(self):
        return self.__epsilon

    def _setEpsilon(self, eps):
        self.__epsilon = eps

    def getPX(self):
        return self.__p_x

    def _setPX(self, p_x):
        self.__p_x = p_x

    def calculateSumX(self, degree):
        return sum([x ** degree for x in self.__x_array])

    def calculateSumY(self):
        return sum(self.__y_array)

    def calculateSumXY(self, degree):
        return sum([self.__x_array[i] ** degree * self.__y_array[i] for i in range(self.__n)])

    def _calculateEpsilon(self):
        return [self.__p_x[i] - self.__y_array[i] for i in range(self.__n)]

    def _calculateExponentialValues(self, a, b):
        return [a * x ** b for x in self.__x_array]

    def _calculateLogarithmValues(self, a, b):
        return [a * log(x) + b for x in self.__x_array]

    def _calculateExponentValues(self, a, b):
        return [a * exp(b * x) for x in self.__x_array]

    def _calculateLinearValues(self, a, b):
        return [a * x + b for x in self.__x_array]

    def _calculatePolynomialSecondValues(self, a0, a1, a2):
        return [a0 + a1 * x + a2 * x ** 2 for x in self.__x_array]

    def _calculatePolynomialThirdValues(self, a0, a1, a2, a3):
        return [a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 for x in self.__x_array]

    def _deviationMeasure(self):
        return sum([eps ** 2 for eps in self.__epsilon])

    def _standardDeviation(self):
        return sqrt(self._deviationMeasure() / self.__n)

    def _printTable(self):
        print('|  X  |  Y | P(x) | epsilon |')
        for i in range(self.__n):
            print(f"| {self.__x_array[i]} | {self.__y_array[i]} | {self.__p_x[i]} | {self.__epsilon[i]} |")

    def _coefficientCorrelation(self):
        x_mean = self._calculateMeanX()
        y_mean = self._calculateMeanY()
        return sum([(self.__x_array[i] - x_mean) * (self.__y_array[i] - y_mean)
                    for i in range(self.__n)]) / sqrt(
            sum([(self.__x_array[i] - x_mean) ** 2 for i in range(self.__n)]) *
            sum([(self.__y_array[i] - y_mean) ** 2 for i in range(self.__n)]))

    def _calculateMeanX(self):
        return sum(self.__x_array) / self.__n

    def _calculateMeanY(self):
        return sum(self.__y_array) / self.__n