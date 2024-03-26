// #include /plot.kojo
// #include /nn.kojo
// #include neural-net.kojo

cleari()
clearOutput()

val seed = 42
initRandomGenerator(seed)
Engine.getInstance.setRandomSeed(seed)

val m = 3
val c = 1

def f(x: Double) = m * x + c

val xData = range(0, 11)
val yData = xData.map(x => f(x) + randomDouble(-3, 3))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val nepochs = 50
val lr = 0.1f

val model = new NeuralNet()
model.train(xData, yData, nepochs, lr)

val yPreds = model.predict(xData)
addLineToChart(chart, Some(s"epoch-$nepochs"), xData, yPreds)
updateChart(chart)

model.close()

