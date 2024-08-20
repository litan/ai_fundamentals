// #include /nn.kojo
// #include /plot.kojo

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

val xData0 = arrayOfDoubles(1, 21)
val yData0 = xData0 map (x => f(x) + random(-3, 3))

val chart = scatterChart("Regression Data", "X", "Y", xData0, yData0)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val xNormalizer = new StandardScaler()
val yNormalizer = new StandardScaler()

val xData = xNormalizer.fitTransform(xData0)
val yData = yNormalizer.fitTransform(yData0)

val nepochs = 500

ndScoped { use =>
    val model = use(new NonlinearModel)
    model.train(xData, yData)
    updateGraph(model, nepochs)
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
    val LEARNING_RATE: Float = 0.1f
    val nm = NDManager.newBaseManager()

    val hidden = 50

    val w1 = nm.randomNormal(0, 0.1f, Shape(1, hidden), DataType.FLOAT32)
    val b1 = nm.zeros(Shape(hidden))

    val w2 = nm.randomNormal(0, 0.1f, Shape(hidden, 1), DataType.FLOAT32)
    val b2 = nm.zeros(Shape(1))

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
        val x = nm.create(xValues).reshape(Shape(-1, 1))
        val y = nm.create(yValues).reshape(Shape(-1, 1))
        for (epoch <- 1 to nepochs) {
            ndScoped { use =>
                val gc = use(gradientCollector)
                val yPred = modelFunction(x)
                val loss = y.sub(yPred).square().mean()
                if (epoch % 50 == 0) {
                    println(s"Loss -- ${loss.getFloat()}")
                }
                gc.backward(loss)
            }

            ndScoped { _ =>
                params.foreach { p =>
                    p.subi(p.getGradient.mul(LEARNING_RATE))
                    p.zeroGradients()
                }
            }
        }
        x.close(); y.close()
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
