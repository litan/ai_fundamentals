// #include /nn.kojo
// #include /dataframe.kojo
// #include neural-net.kojo

cleari()
clearOutput()

val seed = 42
initRandomGenerator(seed)
Engine.getInstance.setRandomSeed(seed)

val df = readCsv(resolvedPath("../../charting/boston-housing.csv"))

val rooms = df.doubleColumn("RM")
val price = df.doubleColumn("MEDV")

val xData = rooms.asDoubleSeq
val yData = price.asDoubleSeq

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val nepochs = 500
val lr = 0.1f

val model = new NeuralNet(25, 15)
model.train(xData, yData, nepochs, lr)
updateGraph(model, nepochs)
model.close()

def updateGraph(model: NeuralNet, n: Int) {
    // take a look at model predictions at the training points
    // and also between the training points
    val xs = xData.flatMap(x => Array(x, x + 0.5)).sorted
    val yPreds = model.predict(xs)
    addLineToChart(chart, Some(s"epoch-$n"), xs, yPreds)
    updateChart(chart)
}
