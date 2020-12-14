import unittest
import algebra.matrix as matrix


class MyTestCase(unittest.TestCase):
    def test_tridiagMatrix1(self):
        a = [4, 9, 3, 3]
        b = [3, 1, 1, 0]
        c = [0, 2, 2, 3]
        f = [2, 3, 3, 2]

        res = [0.4038, 0.1282, 1.038, -0.3718]

        for y1, y2 in zip(matrix.tridiagonalMatrix(a, b, c, f), res):
            self.assertAlmostEqual(y1, y2, delta=1e-3)

    def test_tridiagMatrix2(self):
        a = [4, 4, 6, 3]
        b = [1, 1, 4, 0]
        c = [0, 3, 2, 1]
        f = [1, 1, 1, 1]

        res = [0.2215, 0.1139, -0.1203, 0.3734]

        for y1, y2 in zip(matrix.tridiagonalMatrix(a, b, c, f), res):
            self.assertAlmostEqual(y1, y2, delta=1e-3)

    def test_tridiagMatrix3(self):
        a = [5, 6, 6, 4, 3]
        b = [1, 3, 2, 1, 0]
        c = [0, 2, 1, -2, -1]
        f = [1, 2, 3, 4, 5]

        res = [0.1703, 0.1484, 0.2563, 0.6568, 1.886]
        for y1, y2 in zip(matrix.tridiagonalMatrix(a, b, c, f), res):
            self.assertAlmostEqual(y1, y2, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
