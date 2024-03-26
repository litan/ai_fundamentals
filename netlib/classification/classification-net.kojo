// #include /nn.kojo
import ai.djl.training.dataset.RandomAccessDataset

def range(start: Int, end: Int, step: Double = 1.0): Array[Double] = {
    rangeTill(start, end, step).map(_.toDouble).toArray
}

class ClassificationNet(nDims: Int*) extends AutoCloseable {
    require(nDims.length >= 3, "ClassificationNet problem – you need at least one input, one hidden, and one output layer")
    require(nDims.last > 1, "ClassificationNet problem – you need at least two output classes")
    val nm = NDManager.newBaseManager()

    val nDimsList = nDims.toList
    var inputDim = nDimsList.head

    val params = ArrayBuffer.empty[NDArray]
    val wInitMean = 0f
    val wInitStd = 0.1f

    for (pairList <- nDimsList.sliding(2)) {
        val n1 = pairList(0); val n2 = pairList(1)
        val wn = nm.randomNormal(wInitMean, wInitStd, Shape(n1, n2), DataType.FLOAT32)
        val bn = nm.zeros(Shape(n2))
        params.append(wn); params.append(bn)
    }

    val softmax = new SoftmaxCrossEntropyLoss()

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
        trainingSet: RandomAccessDataset, valSet: RandomAccessDataset,
        epochs: Int, learningRate: Int => Float): Unit = {
        for (epoch <- 1 to epochs) {
            var eloss = 0f
            trainingSet.getData(nm).asScala.foreach { batch0 =>
                ndScoped { use =>
                    val batch = use(batch0)
                    val gc = gradientCollector
                    val x = batch.getData.head.reshape(Shape(-1, 784))
                    val y = batch.getLabels.head
                    val yPred = use(modelFunction(x))
                    val loss = use(softmax.evaluate(new NDList(y), new NDList(yPred)))
                    eloss = loss.getFloat()
                    gc.backward(loss)
                    gc.close()

                    params.foreach { p =>
                        p.subi(p.getGradient.mul(learningRate(epoch)))
                        p.zeroGradients()
                    }
                }
            }
            println(s"[$epoch] Loss -- $eloss")
            //            lossChart.update(epoch, eloss)
        }
        println("Training Done")
    }

    def showAccuracy(valSet: RandomAccessDataset) {
        println("Determining accuracy on the given test set")
        var total = 0l
        var totalGood = 0l
        valSet.getData(nm).asScala.foreach { batch0 =>
            ndScoped { use =>
                val batch = use(batch0)
                val x = batch.getData.head.reshape(Shape(-1, 784))
                val y = batch.getLabels.head
                val yPred = use(modelFunction(x).softmax(1).argMax(1))
                val matches = use(y.toType(DataType.INT64, false).eq(yPred))
                total += matches.getShape.get(0)
                totalGood += matches.countNonzero.getLong()
            }
        }
        val acc = 1f * totalGood / total
        println(acc)
    }

    def save() {
        import java.io._
        val modelFile = s"${kojoCtx.baseDir}/mnist.djl.model"
        println(s"Saving model in file - $modelFile")
        managed { use =>
            val dos = use(new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(modelFile))
            ))
            dos.writeChar('P')
            params.foreach { p =>
                dos.write(p.encode())
            }
        }

    }

    def close() {
        println("Closing remaining ndarrays...")
        params.foreach(_.close())
        nm.close()
        println("Done")
    }
}
