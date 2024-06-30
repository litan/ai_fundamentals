// #include /nn.kojo
// #include /plot.kojo
// #include neural-net.kojo

cleari()
clearOutput()

val seed = 42
initRandomGenerator(seed)
Engine.getInstance.setRandomSeed(seed)

val a = 2
val b = 3
val c = 10
val den = 5000

def f(x: Double) = (a * x * x * x * x + b * x + c) / den

val xData = range(0, 21)
val yData = xData map (x => f(x) + random(-3, 3))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val nepochs = 500
val lr = 0.1f

val model = new NeuralNet(100)
model.train(xData, yData, nepochs, lr)
updateGraph(model, nepochs)
model.close()

def updateGraph(model: NeuralNet, n: Int) {
    // take a look at model predictions at the training points
    // and also between the training points
    val xs = xData.flatMap(x => Array(x, x + 0.5))
    val yPreds = model.predict(xs)
    addLineToChart(chart, Some(s"epoch-$n"), xs, yPreds)
    updateChart(chart)
}
