// #include /nn.kojo
// #include /plot.kojo
// #include neural-net.kojo

val seed = 42
initRandomGenerator(seed)
Engine.getInstance.setRandomSeed(seed)

cleari()
clearOutput()

val m1 = 1
val c1 = -30
val m2 = 6
val c2 = -60
val m3 = -1
val c3 = 30

val xData1 = arrayOfDoubles(1, 7)
val yData1 = xData1.map(x => m1 * x + c1 + randomDouble(-1, 1))

val xData2 = arrayOfDoubles(7, 13)
val yData2 = xData2.map(x => m2 * x + c2 + randomDouble(-1, 1))

val xData3 = arrayOfDoubles(13, 19)
val yData3 = xData3.map(x => m3 * x + c3 + randomDouble(-1, 1))

val xData = xData1 ++ xData2 ++ xData3
val yData = yData1 ++ yData2 ++ yData3

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val nepochs = 30000
val lr = 0.003f

val model = new NeuralNet(4)
model.train(xData, yData, nepochs / 2, lr)
updateGraph(model, nepochs / 2)
model.train(xData, yData, nepochs / 2, lr)
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
