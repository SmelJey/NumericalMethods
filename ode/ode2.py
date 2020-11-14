def euler(a, b, y0, z0, yf, zf, n):
    h = (b - a) / n
    curX = a
    curY = y0
    curZ = z0

    x = [a]
    y = [y0]
    z = [z0]

    for i in range(n):
        newZ = curZ + h * zf(curX, curY, curZ)
        newY = curY + h * yf(curX, curY, curZ)
        curX += h
        curZ = newZ
        curY = newY
        x.append(curX)
        y.append(curY)
        z.append(curZ)

    return x, y, z


def rungeKutta(a, b, y0, z0, yf, zf, n):
    h = (b - a) / n
    curX = a
    curY = y0
    curZ = z0

    x = [a]
    y = [y0]
    z = [z0]

    for i in range(n):
        k1 = yf(curX, curY, curZ)
        m1 = zf(curX, curY, curZ)

        k2 = yf(curX + h / 2, curY + h * k1 / 2, curZ + h * m1 / 2)
        m2 = zf(curX + h / 2, curY + h * k1 / 2, curZ + h * m1 / 2)

        k3 = yf(curX + h / 2, curY + h * k2 / 2, curZ + h * m2 / 2)
        m3 = zf(curX + h / 2, curY + h * k2 / 2, curZ + h * m2 / 2)

        k4 = yf(curX + h, curY + h * k3, curZ + h * m3)
        m4 = zf(curX + h, curY + h * k3, curZ + h * m3)

        curX += h
        curY += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        curZ += h * (m1 + 2 * m2 + 2 * m3 + m4) / 6

        x.append(curX)
        y.append(curY)
        z.append(curZ)

    return x, y, z
