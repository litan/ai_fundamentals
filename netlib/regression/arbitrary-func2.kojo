// #include /plot.kojo
// #include /nn.kojo
// #include neural-net.kojo

cleari()
clearOutput()

val seed = 42
initRandomGenerator(seed)
Engine.getInstance.setRandomSeed(seed)

def f(x: Double) = {
    val table = Map[Int, Double](
        0 -> 20,
        1 -> 18,
        2 -> 16,
        3 -> 17,
        4 -> 19,
        5 -> 20,
        6 -> 18,
        7 -> 16,
        8 -> 14,
        9 -> 16,
        10 -> 20
    )

    table(x.toInt)
}

val xData = arrayOfDoubles(0, 11)
val yData = xData.map(x => f(x))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val nepochs = 500
val lr = 0.01f

val model = new NeuralNet(25, 25)
model.train(xData, yData, nepochs, lr)
updateGraph(model, nepochs)
model.close()

def updateGraph(model: NeuralNet, n: Int) {
    // take a look at model predictions at the training points
    // and also between the training points
    val xs = xData.flatMap(x => Array(x, x + 0.5))
    val yPreds = model.predict(xs)
    addLineToChart(chart, Some(s"epoch-$n"), xs.map(_.toDouble), yPreds.map(_.toDouble))
    updateChart(chart)
}
