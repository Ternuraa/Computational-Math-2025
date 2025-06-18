import numpy as np
from scipy import integrate


class ImproperIntegralSolver:
    """
    Класс для вычисления несобственных интегралов 2 рода
    с использованием различных численных методов
    """

    def __init__(self, func, interval, singularity_type):
        self.func = func
        self.a, self.b = interval
        self.singularity_type = singularity_type
        self.convergent = False

    def _check_convergence(self):
        """Проверка сходимости интеграла"""
        try:
            if self.singularity_type == 'a':
                test = (1e-12) ** 0.5 * self.func(self.a + 1e-12)
            elif self.singularity_type == 'b':
                test = (1e-12) ** 0.5 * self.func(self.b - 1e-12)
            else:
                mid = (self.a + self.b) / 2
                left = ImproperIntegralSolver(self.func, (self.a, mid), 'b')
                right = ImproperIntegralSolver(self.func, (mid, self.b), 'a')
                return left.check_convergence() and right.check_convergence()
            self.convergent = np.isfinite(test)
            return self.convergent
        except:
            self.convergent = False
            return False

    def integrate(self, method='simpson', n=100000):
        """
        Основной метод вычисления интеграла
        :param method: метод интегрирования ('left_rect', 'mid_rect', 'right_rect', 'trapezoid', 'simpson')
        :param n: количество интервалов разбиения
        """
        if not self._check_convergence():
            print("Интеграл не существует")
            return None

        try:
            if self.singularity_type == 'internal':
                return self._handle_internal_singularity(method, n)

            if method in ['left_rect', 'mid_rect', 'right_rect']:
                return self._rectangle_method(method, n)
            elif method == 'trapezoid':
                return self._trapezoid_method(n)
            elif method == 'simpson':
                return self._simpson_method(n)
            else:
                raise ValueError("Неизвестный метод интегрирования")
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            return None

    def _generate_partition(self, n):
        """Генерация разбиения с учётом особенности"""
        if self.singularity_type == 'a':
            return np.linspace(self.a + 1e-12, self.b, n + 1)
        elif self.singularity_type == 'b':
            return np.linspace(self.a, self.b - 1e-12, n + 1)
        else:
            mid = (self.a + self.b) / 2
            return np.concatenate([
                np.linspace(self.a, mid - 1e-12, n // 2 + 1),
                np.linspace(mid + 1e-12, self.b, n // 2 + 1)[1:]
            ])

    def _rectangle_method(self, method, n):
        """Метод прямоугольников"""
        x = self._generate_partition(n)
        h = (x[-1] - x[0]) / n

        if method == 'left_rect':
            points = x[:-1]
        elif method == 'right_rect':
            points = x[1:]
        else:  # mid_rect
            points = (x[:-1] + x[1:]) / 2

        y = self.func(points)
        return h * np.sum(y)

    def _trapezoid_method(self, n):
        """Метод трапеций"""
        x = self._generate_partition(n)
        y = self.func(x)
        return np.trapezoid(y, x)

    def _simpson_method(self, n):
        """Метод Симпсона"""
        if n % 2 != 0:
            n += 1  # Делаем четное количество интервалов
        x = self._generate_partition(n)
        y = self.func(x)
        h = (x[-1] - x[0]) / n
        return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

    def _handle_internal_singularity(self, method, n):
        """Обработка внутренней особенности"""
        c = (self.a + self.b) / 2
        left = ImproperIntegralSolver(self.func, (self.a, c), 'b')
        right = ImproperIntegralSolver(self.func, (c, self.b), 'a')
        return left.integrate(method, n // 2) + right.integrate(method, n // 2)


# Примеры использования
if __name__ == "__main__":
    # Тестовые интегралы
    tests = [
        ("∫1/√x dx от 0 до 1", lambda x: 1 / np.sqrt(x), (0, 1), 'a'),
        ("∫1/√(1-x) dx от 0 до 1", lambda x: 1 / np.sqrt(1 - x), (0, 1), 'b'),
        ("∫1/|x-0.5|^(1/3) dx от 0 до 1", lambda x: 1 / np.abs(x - 0.5) ** (1 / 3), (0, 1), 'internal'),
        ("∫1/x dx от 0 до 1 (расходится)", lambda x: 1 / x, (0, 1), 'a')

    ]

    methods = ['left_rect', 'mid_rect', 'right_rect', 'trapezoid', 'simpson']

    for desc, func, interval, stype in tests:
        print(f"\n{desc}")
        solver = ImproperIntegralSolver(func, interval, stype)

        for method in methods:
            result = solver.integrate(method, n=100000)
            if result is not None:
                print(f"{method:10} → {result:.6f}")

        # Проверка расходимости
        if not solver._check_convergence():
            print("Результат: Интеграл не существует")