import unittest
import math
import analysis.integrals as integral


def f(x):
    return x * x + math.log(x) - 4


def antiderivative(x):
    return x**3 / 3 - 5 * x + x * math.log(x)


a = 1.5
b = 2.0

actualValue = antiderivative(b) - antiderivative(a)


class MyTestCase(unittest.TestCase):
    def test_middleRectangles(self):
        eps = 1e-8
        res = integral.integrate(f, a, b, 3, eps, method=integral.middleRectangles)
        self.assertAlmostEqual(res, actualValue, delta=eps)

    def test_newtonCotes(self):
        eps = 1e-8
        res = integral.integrate(f, a, b, 3, eps, method=integral.newtonCotes3)
        self.assertAlmostEqual(res, actualValue, delta=eps)

    def test_chebyshev(self):
        eps = 1e-8
        res = integral.integrate(f, a, b, 3, eps, method=integral.chebyshev3)
        self.assertAlmostEqual(res, actualValue, delta=eps)

    def test_monteKarlo(self):
        eps = 1e-3
        res = integral.integrate(f, a, b, 3, eps, method=integral.monteKarlo)
        self.assertAlmostEqual(res, actualValue, delta=eps * 10)


if __name__ == '__main__':
    unittest.main()
