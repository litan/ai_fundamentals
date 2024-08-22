// #include /plot.kojo
// #include /nn.kojo

cleari()
clearOutput()

val seed = 42
initRandomGenerator(seed)
Engine.getInstance.setRandomSeed(seed)

val m = 3
val c = 1

def f(x: Double) = m * x + c

val xData = arrayOfDoubles(0, 11)
val yData = xData.map(x => f(x) + randomDouble(-3, 3))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val nepochs = 500

ndScoped { use =>
    val model = use(new Model)
    model.train(xData, yData)
    updateGraph(model, nepochs)
}

def updateGraph(model: Model, n: Int) {
    // take a look at model predictions at the training points
    // and also between the training points
    val xs = xData.flatMap(x => Array(x, x + 0.5))
    val yPreds = model.predict(xs)
    addLineToChart(chart, Some(s"epoch-$n"), xs, yPreds)
    updateChart(chart)
}

class Model extends AutoCloseable {
    val LEARNING_RATE: Float = 0.005f
    val nm = ndMaker

    val w = nm.create(1f).reshape(Shape(1, 1))
    val b = nm.create(0f).reshape(Shape(1))

    val params = new NDList(w, b).asScala

    params.foreach { p =>
        p.setRequiresGradient(true)
    }

    def modelFunction(x: NDArray): NDArray = {
        x.matMul(w).add(b)
    }

    def train(xValuesD: Array[Double], yValuesD: Array[Double]): Unit = {
        val xValues = xValuesD.map(_.toFloat)
        val yValues = yValuesD.map(_.toFloat)

        ndScoped { _ =>
            val x = nm.create(xValues).reshape(Shape(-1, 1))
            val y = nm.create(yValues).reshape(Shape(-1, 1))
            for (epoch <- 1 to nepochs) {
                ndScoped { _ =>
                    val gc = gradientCollector
                    val yPred = modelFunction(x)
                    val loss = y.sub(yPred).square().mean()
                    gc.backward(loss)
                    gc.close()

                    for (p <- params) {
                        p.subi(p.getGradient.mul(LEARNING_RATE))
                        p.zeroGradients()
                    }
                }
            }
        }
        println("Training Done")
        println(w)
        println(b)
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
        params.foreach { p =>
            p.close()
        }
        nm.close()
        println("Done")
    }
}
