// #include /dataframe.kojo
// #include /nn.kojo
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

val nepochs = 50
val lr = 0.1f

val model = new NeuralNet()
model.train(xData, yData, nepochs, lr)

val yPreds = model.predict(xData)
addLineToChart(chart, Some(s"epoch-$nepochs"), xData, yPreds)
updateChart(chart)

model.close()

