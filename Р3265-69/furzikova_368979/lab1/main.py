import numpy as np

# Функция для округления чисел
def round_number(num, digits=3):
    return f'{num:.{digits}f}'

# Функция для чтения системы уравнений из файла
def read_from_file(file_path):
    try:
        n = 0
        equations = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Пропускаем пустые строки
                    n += 1
        if n > 20:
            print('Файл содержит более 20 уравнений. Пожалуйста, уменьшите количество уравнений и попробуйте снова.')
            return
        with open(file_path, 'r', encoding='utf-8') as file:
            for row in file:
                if row.strip():  # Пропускаем пустые строки
                    parts = row.split()
                    if parts[-2] != '|' or len(parts) - 2 != n:
                        print('Ошибка формата файла. Пожалуйста, проверьте формат и попробуйте снова.')
                        return
                    equations.append(parts)
        solver = EquationSolver(n, prepare_equations(equations, n))
        solver.solve()
    except FileNotFoundError:
        print('Файл не найден по указанному пути:', file_path)

# Функция для ввода системы уравнений вручную
def input_manually():
    try:
        n = int(input('Введите количество уравнений (не более 20): '))
        if 1 < n <= 20:
            equations = []
            print('Введите коэффициенты уравнений в формате:')
            print('\t', 'a1 a2 ... an | b')
            for i in range(n):
                while True:
                    line = input(f'{i + 1}: ').split()
                    if len(line) - 2 != n or line[-2] != '|':
                        print('Количество коэффициентов не соответствует количеству уравнений или неправильный формат.')
                        print('Пожалуйста, попробуйте снова.')
                    else:
                        equations.append(line)
                        break
            solver = EquationSolver(n, prepare_equations(equations, n))
            solver.solve()
        else:
            print('Некорректный ввод! Количество уравнений должно быть от 2 до 20.')
    except ValueError:
        print('Некорректный ввод! Пожалуйста, введите целое число.')

# Преобразуем строки в числа
def prepare_equations(arr, n):
    for i in range(n):
        for j in range(n):
            arr[i][j] = float(arr[i][j])
        arr[i][-1] = float(arr[i][-1])
    return arr

# Решение системы уравнений с использованием numpy
def solve_with_numpy(equations, n):
    # Преобразуем уравнения в матрицу коэффициентов и вектор свободных членов
    A = np.array([equations[i][:n] for i in range(n)], dtype=float)
    B = np.array([equations[i][-1] for i in range(n)], dtype=float)
    
    # Решаем систему уравнений
    solutions = np.linalg.solve(A, B)
    
    # Вычисляем определитель матрицы
    determinant = np.linalg.det(A)
    
    return solutions, determinant

class EquationSolver:
    def __init__(self, n, equations):
        self.n = n
        self.original_equations = [row.copy() for row in equations]  # Сохраняем исходную матрицу
        self.equations = equations                                   # Матрица для преобразований
        self.solutions = []
        self.swaps = 0

    def solve(self):
        try:
            print('\nИсходная система уравнений:')
            self.print_equations()

            self.convert_to_triangle()
            print('\nТреугольная матрица:')
            self.print_equations()

            print('\nКоличество перестановок:', self.swaps)

            self.calculate_determinant()

            self.find_solutions()
            self.print_solutions()

            self.print_residuals()

            # Передаем исходную матрицу в numpy, а не модифицированную
            numpy_solutions, numpy_determinant = solve_with_numpy(self.original_equations, self.n)
            print('\nРешения системы через numpy:')
            for i in range(self.n):
                print(f'\tx[{i}] = {round_number(numpy_solutions[i])}')
            print(f'\nОпределитель матрицы через numpy: {round_number(numpy_determinant)}')

            # Сравнение результатов
            print('\nСравнение результатов:')
            print(f'\tРазница в решениях: {np.linalg.norm(np.array(self.solutions) - numpy_solutions)}')
            print(f'\tРазница в определителях: {abs(self.determinant - numpy_determinant)}')

        except ZeroDivisionError:
            print('Ошибка: деление на ноль!')
        except ArithmeticError:
            print('Ошибка: система не имеет решений!')
        except np.linalg.LinAlgError as e:
            print(f'Ошибка при решении системы через numpy: {e}')

    def check_diagonal(self, i):
        for j in range(i, self.n):
            if self.equations[j][i] != 0:
                self.equations[i], self.equations[j] = self.equations[j], self.equations[i]
                self.swaps += 1
                return
        print('Система не имеет решений!')
        raise ArithmeticError

    def convert_to_triangle(self):
        for i in range(self.n):
            if self.equations[i][i] == 0:
                self.check_diagonal(i)
            for m in range(i + 1, self.n):
                factor = -(self.equations[m][i] / self.equations[i][i])
                for j in range(i, self.n):
                    self.equations[m][j] += factor * self.equations[i][j]
                self.equations[m][-1] += factor * self.equations[i][-1]

    def print_equations(self):
        for i in range(self.n):
            equation = ' '.join([f'{round_number(self.equations[i][j])} * x[{j}]' for j in range(self.n)])
            print(f'{equation} | {round_number(self.equations[i][-1])}')

    def calculate_determinant(self):
        self.determinant = 1
        for i in range(self.n):
            self.determinant *= self.equations[i][i]
        if self.swaps % 2 == 1:
            self.determinant *= -1
        print(f'\nОпределитель матрицы: {round_number(self.determinant)}\n')
        if self.determinant == 0:
            print('Система вырождена и не имеет решений.')
            raise ArithmeticError

    def find_solutions(self):
        self.solutions = [0] * self.n
        for i in range(self.n - 1, -1, -1):
            self.solutions[i] = self.equations[i][-1] / self.equations[i][i]
            for j in range(i - 1, -1, -1):
                self.equations[j][-1] -= self.equations[j][i] * self.solutions[i]

    def print_solutions(self):
        print('Решения системы:')
        for i in range(self.n):
            print(f'\tx[{i}] = {round_number(self.solutions[i])}')

    def print_residuals(self):
        print('\nВеличины невязок:')
        for i in range(self.n):
            residual = sum(self.equations[i][j] * self.solutions[j] for j in range(self.n)) - self.equations[i][-1]
            print(f'\tНевязка для уравнения {i + 1}: {round_number(abs(residual))}')

# Основное меню программы
def main():
    print('Решение систем линейных уравнений методом Гаусса')

    while True:
        try:
            print('\nДоступные действия:')
            print('\t1: Загрузить систему уравнений из файла.')
            print('\t2: Ввести систему уравнений вручную.')
            print('\t3: Выйти из программы.')
            choice = int(input('Выберите действие: '))

            if choice == 1:
                print('Загрузка системы уравнений из файла.')
                print('Формат файла должен быть следующим (максимум 20 уравнений):')
                print('\ta11 a12 ... a1n | b1')
                print('\ta21 a22 ... a2n | b2')
                print('\t... ... ... ... | ..')
                print('\tan1 an2 ... ann | bn')
                file_path = input('Введите путь к файлу: ').strip()
                read_from_file(file_path)
            elif choice == 2:
                print('Ручной ввод системы уравнений.')
                input_manually()
            elif choice == 3:
                print('Программа завершена. До свидания!')
                break
            else:
                print('Некорректный выбор. Пожалуйста, выберите действие от 1 до 3.')
        except KeyboardInterrupt:
            print('\nПрограмма была прервана пользователем.')
        except Exception as e:
            print(f'Произошла ошибка: {e}')

if __name__ == '__main__':
    main()