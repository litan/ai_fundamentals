// #include /plot.kojo

cleari()
clearOutput()

val m = 3
val c = 1

def f(x: Double) = m * x + c

repeatFor(0 to 10) { x =>
    val y = f(x)
    println(s"x=$x, f(x)=$y")
}

val xData = arrayOfDoubles(0, 11)
val yData = xData.map(x => f(x))

val chart = scatterChart("Linear Function", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)
