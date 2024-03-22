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

val xData = Array.tabulate(11)(n => n.toDouble)
val yData = xData.map(x => f(x) + randomDouble(-3, 3))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val nepochs = 10
val lr = 0.99f

ndScoped { use =>
    val model = use(new NeuralNet())
    model.train(xData, yData, nepochs, lr)
    updateGraph(model, nepochs)
}

def updateGraph(model: NeuralNet, n: Int) {
    // take a look at model predictions at the training points
    // and also between the training points
    val xs = xData.flatMap(x => Array(x, x + 0.5))
    val yPreds = model.predict(xs)
    addLineToChart(chart, Some(s"epoch-$n"), xs.map(_.toDouble), yPreds.map(_.toDouble))
    updateChart(chart)
}
