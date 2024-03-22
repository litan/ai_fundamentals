// #include /plot.kojo

cleari()
clearOutput()

val m = 3
val c = 1

def f(x: Double) = m * x + c

val xData = Array.tabulate(11)(n => n.toDouble)
val yData = xData.map(x => f(x) + randomDouble(-3, 3))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)
