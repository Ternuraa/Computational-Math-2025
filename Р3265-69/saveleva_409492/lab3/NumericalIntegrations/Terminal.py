from Methods.MethodRectangles import MethodRectangles
from Methods.MethodSympson import MethodSympson
from Methods.MethodTrapezoidal import MethodTrapezoidal
from Validator import Validator
from Exceptions.IncorrectValueException import IncorrectValueException
from Functions import *


class Terminal:

    def work(self):
        try:
            print('\t\t\tЛабораторная работа №3\n\t\t\tЧисленное интегрирование')
            for i in range(len(FUNCTIONS)):
                print(f"{i + 1}. {FUNCTIONS[i + 1]['FUNCTION']}")
            function_number = self.enterFunctionNumber()
            print('\t\tМетоды решения:')
            for i in range(len(METHODS)):
                print(f'{i + 1}.{METHODS[i]}')
            method_number = self.enterFunctionMethod()
            a, b = self.enterBorders()
            epsilon = self.enterEpsilon()
            n = self.enterN()

            if method_number == 1:
                right_rectangles = MethodRectangles(epsilon, a, b, n)
                right_rectangles.iterateRightRectangles(number_of_function=function_number)
            elif method_number == 2:
                left_rectangles = MethodRectangles(epsilon, a, b, n)
                left_rectangles.iterateLeftRectangles(number_of_function=function_number)
            elif method_number == 3:
                middle_rectangles = MethodRectangles(epsilon, a, b, n)
                middle_rectangles.iterateMiddleRectangles(number_of_function=function_number)
            elif method_number == 4:
                trapezoidal_method = MethodTrapezoidal(epsilon, a, b, n)
                trapezoidal_method.iterateTrapezoidalMethod(number_of_function=function_number)
            else:
                sympson_method = MethodSympson(epsilon, a, b, n)
                sympson_method.iterateSympsonlMethod(number_of_function=function_number)
        except IncorrectValueException as e:
            print(e.message)
            return

    def enterFunctionNumber(self):
        try:
            print('Введите номер функции:')
            return Validator.validateFunctionNumber()
        except IncorrectValueException as e:
            print(e.message)
            return self.enterFunctionNumber()

    def enterFunctionMethod(self):
        try:
            print('Введите номер метода:')
            return Validator.validateFunctionMethod()
        except IncorrectValueException as e:
            print(e.message)
            return self.enterFunctionMethod()

    def enterN(self):
        try:
            print('Введите начальное число отрезков для разбиения:')
            n = Validator.validateN()
            return n
        except IncorrectValueException as e:
            print(e.message)
            return self.enterN()

    def enterBorders(self):
        try:
            print('Введите границы интервала a и b:')
            print('a = ', end='')
            a = Validator.validateNumber(input())
            print('b = ', end='')
            b = Validator.validateNumber(input())
            Validator.validateBorders(a, b)
            return a, b
        except IncorrectValueException as e:
            print(e.message)
            return self.enterBorders()

    def enterEpsilon(self):
        try:
            print('Введите точность epsilon:')
            eps = Validator.validateEpsilon(input())
            return eps
        except IncorrectValueException as e:
            print(e.message)
            return self.enterEpsilon()