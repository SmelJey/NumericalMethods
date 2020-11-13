import unittest
import math
import util
import ode.ode1 as ode


def testF1(x, y):
    return 1 / x + 2 * x


def testSol1(x):
    return math.log(x) + x ** 2


def testF2(x, y):
    return x - 2 * y


def testSol2(x):
    return (5 * math.exp(-2 * x) + 2 * x - 1) / 4


class Ode1Test(unittest.TestCase):
    def checkOde(self, method, yf, sol, y0, a, b, n, order):
        h = (b - a) / n
        eps = h ** order
        resX, resY = method(a, b, y0, yf, n)
        expX, expY = util.createPlot(sol, a, b, n)
        diffX = max([abs(x - y) for x, y in zip(resX, expX)])
        diffY = max([abs(x - y) for x, y in zip(resY, expY)])
        print("method:", method.__name__, "error:", diffY, "expected:", eps)

        self.assertEqual(diffX, 0)
        self.assertAlmostEqual(diffY, 0, delta=eps)

    def test_euler_simple(self):
        self.checkOde(ode.euler, testF1, testSol1, 1, 1, 2, 10000, 1)

    def test_euler(self):
        self.checkOde(ode.euler, testF2, testSol2, 1, 0, 2, 10000, 1)

    def test_euler2_simple(self):
        self.checkOde(ode.euler2, testF1, testSol1, 1, 1, 2, 10000, 2)

    def test_euler2(self):
        self.checkOde(ode.euler2, testF2, testSol2, 1, 0, 2, 10000, 2)

    def test_rk_simple(self):
        self.checkOde(ode.rungeKutta, testF1, testSol1, 1, 1, 2, 1000, 4)

    def test_rk(self):
        self.checkOde(ode.rungeKutta, testF2, testSol2, 1, 0, 2, 1000, 4)


if __name__ == '__main__':
    unittest.main()
