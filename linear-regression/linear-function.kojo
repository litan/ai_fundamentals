// #include /plot.kojo

cleari()
clearOutput()

val m = 3
val c = 1

def f(x: Double) = m * x + c

val xData = arrayOfDoubles(0, 11)
val yData = xData.map(x => f(x))

repeatFor(0 to 10) { idx =>
    println(s"x=${xData(idx)}, f(x)=${yData(idx)}")
}

val chart = scatterChart("Linear Function", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)
