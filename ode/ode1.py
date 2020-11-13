def euler(a, b, y0, yf, n):
    h = (b - a) / n
    curX = a
    curY = y0

    x = [a]
    y = [y0]

    for i in range(n):
        curY += h * yf(curX, curY)
        curX += h
        x.append(curX)
        y.append(curY)

    return x, y


def euler2(a, b, y0, yf, n):
    h = (b - a) / n
    curX = a
    curY = y0

    x = [a]
    y = [y0]

    for i in range(n):
        dy = h * yf(curX + h / 2, curY + h / 2 * yf(curX, curY))
        curX += h
        curY += dy
        x.append(curX)
        y.append(curY)

    return x, y


def rungeKutta(a, b, y0, yf, n):
    h = (b - a) / n
    curX = a
    curY = y0

    x = [a]
    y = [y0]

    for i in range(n - 1):
        k1 = yf(curX, curY)
        k2 = yf(curX + h / 2, curY + h * k1 / 2)
        k3 = yf(curX + h / 2, curY + h * k2 / 2)
        k4 = yf(curX + h, curY + h * k3)

        curX += h
        curY += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x.append(curX)
        y.append(curY)

    return x, y
