// #include /nn.kojo

def range(start: Int, end: Int, step: Double = 1.0): Array[Double] = {
    rangeTill(start, end, step).map(_.toDouble).toArray
}

class NeuralNet(numHiddenUnits: Int*) extends AutoCloseable {
    val xNormalizer = new StandardScaler()
    val yNormalizer = new StandardScaler()

    val nm = NDManager.newBaseManager()

    val nhu = numHiddenUnits.toList
    var hidden = 1
    val params = ArrayBuffer.empty[NDArray]
    val wInitMean = 0.1f
    val wInitStd = 0.3f
    val bInitMean = 0.1f
    val bInitStd = 0.3f

    if (nhu.nonEmpty) {
        hidden = nhu.head
        val restHidden = nhu.tail

        val w1 = nm.randomNormal(wInitMean, wInitStd, Shape(1, hidden), DataType.FLOAT32)
        val b1 = nm.randomNormal(bInitMean, bInitStd, Shape(hidden), DataType.FLOAT32)
        params.append(w1); params.append(b1)

        for (h <- restHidden) {
            val wn = nm.randomNormal(wInitMean, wInitStd, Shape(hidden, h), DataType.FLOAT32)
            val bn = nm.randomNormal(bInitMean, bInitStd, Shape(h), DataType.FLOAT32)
            hidden = h
            params.append(wn); params.append(bn)
        }
    }

    val wf = nm.randomNormal(wInitMean, wInitStd, Shape(hidden, 1), DataType.FLOAT32)
    val bf = nm.randomNormal(bInitMean, bInitStd, Shape(1), DataType.FLOAT32)
    params.append(wf); params.append(bf)

    //    val params = new NDList(w1, b1, w2, b2).asScala

    params.foreach { p =>
        p.setRequiresGradient(true)
    }

    def modelFunction(x: NDArray): NDArray = {
        var lna = x
        for (n <- 0 until (params.length - 2) by 2) {
            lna = lna.matMul(params(n)).add(params(n + 1))
            lna = Activation.relu(lna)
        }
        lna.matMul(params(params.length - 2)).add(params.last)
    }

    def train(
        xValuesRaw: Array[Double], yValuesRaw: Array[Double],
        epochs: Int, lr: Float = 0.01f): Unit = {
        ndScoped { use =>
            val xValues = xNormalizer.fitTransform(xValuesRaw).map(_.toFloat)
            val yValues = yNormalizer.fitTransform(yValuesRaw).map(_.toFloat)
            val x = nm.create(xValues).reshape(Shape(-1, 1))
            val y = nm.create(yValues).reshape(Shape(-1, 1))
            for (epoch <- 1 to epochs) {
                ndScoped { use =>
                    val gc = use(gradientCollector)
                    val yPred = modelFunction(x)
                    val loss = y.sub(yPred).square().mean()
                    if (epoch % 5 == 0) {
                        println(s"[Epoch $epoch] Loss – ${loss.getFloat()}")
                    }
                    gc.backward(loss)
                }

                ndScoped { _ =>
                    params.foreach { p =>
                        p.subi(p.getGradient.mul(lr))
                        p.zeroGradients()
                    }
                }
            }
            println("Training Done")
        }
    }

    def predict(xValuesRaw: Array[Double]): Array[Double] = {
        val xValues = xNormalizer.fitTransform(xValuesRaw).map(_.toFloat)
        ndScoped { _ =>
            val x = nm.create(xValues).reshape(Shape(-1, 1))
            val y = modelFunction(x)
            yNormalizer.inverseTransform(y.toFloatArray.map(_.toDouble))
        }
    }

    def close() {
        println("Closing remaining ndarrays...")
        params.foreach(_.close())
        nm.close()
        println("Done")
    }
}
