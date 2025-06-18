from Exceptions.IncorrectValueException import IncorrectValueException

AMOUNT_OF_INPUT_ARRAYS = 2
MIN_DOTS = 8
MAX_DOTS = 12


class Validator:

    @staticmethod
    def validateNumber(number: str):
        try:
            number = float(number)
            return number
        except ValueError:
            raise IncorrectValueException(f'Необходимо ввести число.')


    @staticmethod
    def validateArraySize(arr):
        length = len(arr)
        if MIN_DOTS > length or length > MAX_DOTS:
            raise IncorrectValueException('Размер массива должен находится в пределах [8;12].')

    @staticmethod
    def validateEqualityLengths(arr_1, arr_2):
        if len(arr_1) != len(arr_2):
            raise IncorrectValueException('Длины массивов не равны.')