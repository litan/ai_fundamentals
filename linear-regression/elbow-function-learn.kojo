// #include /nn.kojo
// #include /plot.kojo

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

val xData0 = xData1 ++ xData2 ++ xData3
val yData0 = yData1 ++ yData2 ++ yData3

val chart = scatterChart("Regression Data", "X", "Y", xData0, yData0)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val xNormalizer = new StandardScaler()
val yNormalizer = new StandardScaler()

val xData = xNormalizer.fitTransform(xData0)
val yData = yNormalizer.fitTransform(yData0)

ndScoped { use =>
    val model = use(new NonlinearModel)
    model.train(xData, yData)
}

def updateGraph(model: NonlinearModel, n: Int) {
    // take a look at model predictions at the training points
    // and also between the training points
    val xs = xData.flatMap(x => Array(x, x + 0.1))
    val yPreds = model.predict(xs)
    val yPreds0 = yNormalizer.inverseTransform(yPreds)
    val xs0 = xNormalizer.inverseTransform(xs)
    addLineToChart(chart, Some(s"epoch-$n"), xs0, yPreds0)
    updateChart(chart)
}

class NonlinearModel extends AutoCloseable {
    val LEARNING_RATE: Float = 0.003f
    val nm = NDManager.newBaseManager()

    val hidden = 4

    val w1 = nm.randomNormal(0.1f, 0.3f, Shape(1, hidden), DataType.FLOAT32)
    val b1 = nm.randomNormal(0.1f, .3f, Shape(hidden), DataType.FLOAT32)

    val w2 = nm.randomNormal(0.1f, 0.3f, Shape(hidden, 1), DataType.FLOAT32)
    val b2 = nm.randomNormal(0.1f, .3f, Shape(1), DataType.FLOAT32)

    val params = new NDList(w1, b1, w2, b2).asScala

    params.foreach { p =>
        p.setRequiresGradient(true)
    }

    def modelFunction(x: NDArray): NDArray = {
        val l1 = x.matMul(w1).add(b1)
        val l1a = Activation.relu(l1)
        l1a.matMul(w2).add(b2)
    }

    def train(xValuesD: Array[Double], yValuesD: Array[Double]): Unit = {
        val xValues = xValuesD.map(_.toFloat)
        val yValues = yValuesD.map(_.toFloat)
        ndScoped { _ =>
            val x = nm.create(xValues).reshape(Shape(-1, 1))
            val y = nm.create(yValues).reshape(Shape(-1, 1))
            for (epoch <- 1 to 30000) {
                ndScoped { _ =>
                    val gc = gradientCollector
                    val yPred = modelFunction(x)
                    val loss = y.sub(yPred).square().mean()
                    if (epoch % 1000 == 0) {
                        println(s"Loss($epoch) -- ${loss.getFloat()}")
                    }
                    gc.backward(loss)
                    gc.close()

                    for (p <- params) {
                        p.subi(p.getGradient.mul(LEARNING_RATE))
                        p.zeroGradients()
                    }
                }

                if (epoch % 15000 == 0) {
                    updateGraph(this, epoch)
                }
            }
        }
        println("Training Done")
    }

    def predict(xValuesD: Array[Double]): Array[Double] = {
        val xValues = xValuesD.map(_.toFloat)
        ndScoped { _ =>
            val x = nm.create(xValues).reshape(Shape(-1, 1))
            val y = modelFunction(x)
            y.toFloatArray.map(_.toDouble)
        }
    }

    def close() {
        println("Closing remaining ndarrays...")
        params.foreach(_.close())
        nm.close()
        println("Done")
    }
}
