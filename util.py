def createPlot(f, a, b, n):
    h = (b - a) / n
    curX = a
    x = [curX]
    y = [f(curX)]
    for i in range(n):
        curX += h
        x.append(curX)
        y.append(f(curX))
    return x, y
