import numexpr as ne
from sympy import *

x = symbols('x')

METHODS = ['Метод правых прямоугольников',
           'Метод левых прямоугольников',
           'Метод средних прямоугольников',
           'Метод трапеций',
           'Метод Симпсона']

FUNCTIONS = {
    1:
        {
            'FUNCTION': 'f(x)=sin(x)+cos(x)'
        },
    2:
        {
            'FUNCTION': 'f(x)=19*x**5-67*x**2+100*x-191'
        },
    3:
        {
            'FUNCTION': 'f(x)=5**x+5*x'
        },
    4:
        {
            'FUNCTION': 'f(x)=(cos(x))**2'
        },
    5:
        {
            'FUNCTION': 'f(x)=x**2'
        },
6:
        {
            'FUNCTION': 'f(x)=5*x**3 - 2*x**2 + 3*x - 15'
        }

}


def calculateFunction(number_of_function, x: float) -> float:
    return ne.evaluate(FUNCTIONS[number_of_function]['FUNCTION'].split('=')[1], local_dict={'x': x})
