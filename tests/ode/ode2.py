import unittest
import math
import util
import ode.ode2 as ode


def zfTest(x, y, z):
    return x - y - 2 * z


def solTest(x):
    return x + math.exp(-x) * (x + 2) - 2


def zfTest2(x, y, z):
    return -math.sinh(x) * z - x * y - 1


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
        self.checkOde2(ode.euler, lambda x, y, z: z, zfTest, solTest, 0, 0, 0, 1, 1000, 1)

    def test_rk(self):
        self.checkOde2(ode.rungeKutta, lambda x, y, z: z, zfTest, solTest, 0, 0, 0, 1, 100, 4)

    def test_bp_shooting(self):
        x, y, y0 = ode.shootingMethodY(0, 1, 2, 2, zfTest2, 1000, 1e-5)
        self.assertAlmostEqual(y0, 0.8666736566040598, delta=1e-5)

    def test_bp_secant(self):
        x, y, y0 = ode.secantMethodY(0, 1, 2, 2, zfTest2, 1000)
        self.assertAlmostEqual(y0, 0.8666736566040598, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
