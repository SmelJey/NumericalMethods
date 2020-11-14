import unittest
import math
import util
import ode.ode2 as ode


def zf(x, y, z):
    return x - y - 2 * z


def sol(x):
    return x + math.exp(-x) * (x + 2) - 2


class Ode2Test(unittest.TestCase):
    def checkOde2(self, method, yf, zf, sol, y0, z0, a, b, n, order):
        h = (b - a) / n
        eps = h ** order
        resX, resY, resZ = method(a, b, y0, z0, yf, zf, n)
        expX, expY = util.createPlot(sol, a, b, n)
        diffX = max([abs(x - y) for x, y in zip(resX, expX)])
        diffY = max([abs(x - y) for x, y in zip(resY, expY)])
        print("method:", method.__name__, "error:", diffY, "expected:", eps)

        self.assertEqual(diffX, 0)
        self.assertAlmostEqual(diffY, 0, delta=eps)

    def test_euler(self):
        self.checkOde2(ode.euler, lambda x, y, z: z, zf, sol, 0, 0, 0, 1, 1000, 1)

    def test_rk(self):
        self.checkOde2(ode.rungeKutta, lambda x, y, z: z, zf, sol, 0, 0, 0, 1, 100, 4)


if __name__ == '__main__':
    unittest.main()
