from Exceptions.IncorrectValueException import IncorrectValueException
from Functions import FUNCTIONS, METHODS


class Validator:
    MIN_EPSILON = 0

    @staticmethod
    def validateEpsilon(epsilon: str):
        try:
            epsilon = float(epsilon)
            if epsilon >= Validator.MIN_EPSILON:
                return epsilon
            else:
                raise IncorrectValueException(f'Точность (эпсилон) должна быть больше или равна 0.')
        except ValueError:
            raise IncorrectValueException(f'Точность (эпсилон) - число с плавающей точкой.')

    @staticmethod
    def validateNumber(number: str):
        try:
            number = float(number)
            return number
        except ValueError:
            raise IncorrectValueException(f'Необходимо ввести число.')

    @staticmethod
    def validateFunctionNumber():
        equation_number = int(Validator.validateNumber(input()))
        if 0 < equation_number <= len(FUNCTIONS):
            return equation_number
        else:
            raise IncorrectValueException('Неверное введен номер функции. Попробуйте еще раз.')


    @staticmethod
    def validateFunctionMethod():
        equation_method = int(Validator.validateNumber(input()))
        if 0 < equation_method <= len(METHODS):
            return equation_method
        else:
            raise IncorrectValueException('Неверное введен номер метода. Попробуйте еще раз.')

    @staticmethod
    def validateN():
        n = int(Validator.validateNumber(input()))
        if n > 3:
            return n
        else:
            raise IncorrectValueException('Неверное введено значение числа разбиения интервала интегрирования. Число должно быть строго больше 3.. Попробуйте еще раз.')

    @staticmethod
    def validateBorders(border_left: float, border_right: float):
        if border_left < border_right:
            return True
        else:
            raise IncorrectValueException(f'Левая граница должна быть строго меньше правой.')