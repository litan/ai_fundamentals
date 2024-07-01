// #include /nn.kojo
import ai.djl.training.dataset.RandomAccessDataset

def range(start: Int, end: Int, step: Double = 1.0): Array[Double] = {
    rangeTill(start, end, step).map(_.toDouble).toArray
}

trait TrainingListener {
    def onEpochDone(epoch: Int, loss: Double)
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

    var otListener: Option[TrainingListener] = None
    def setTrainingListener(tl: TrainingListener) {
        otListener = Some(tl)
    }

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
            println(s"[Epoch $epoch] Loss – $eloss")
            otListener.foreach(_.onEpochDone(epoch, eloss))
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

    val rad = 20

    def hiddenPicture(r: Int, first: Boolean, last: Boolean) = {
        val fillc = if (first) cm.gray else if (last) cm.blue else cm.green
        Picture.circle(r).withFillColor(fillc).withNoPen()
    }

    def visualize() {
        val pic = netPic
        draw(pic)
    }

    def netPic: Picture = {
        val vgap = 20
        val hgap = 60
        var ldx = 0
        var ldy = 0

        val lineData = ArrayBuffer.empty[ArrayBuffer[Point]]
        var lineDataCurr = ArrayBuffer.empty[Point]

        def vertPics(n: Int, first: Boolean, last: Boolean): Picture = {
            ldy = rad - (n * 2 * rad + (n - 1) * vgap) / 2
            val ab = ArrayBuffer.empty[Picture]
            repeatFor(1 to n) { idx =>
                val pic = hiddenPicture(rad, first, last)
                ab.append(pic)
                lineDataCurr.append(Point(ldx, ldy))
                ldy += vgap + 2 * rad
                if (idx != n) {
                    ab.append(Picture.vgap(vgap))
                }
            }
            picCol(ab)
        }

        val ab = ArrayBuffer.empty[Picture]
        // add hidden pic to anchor the centered row of layer pics
        ab.append(Picture.circle(rad).withNoPen.withTranslation(-2 * rad, 0))
        for ((dim, cnt) <- nDims.zipWithIndex) {
            val first = (cnt == 0)
            val last = (cnt == nDims.length - 1)
            val n = dim
            lineDataCurr = ArrayBuffer.empty[Point]
            val pics = vertPics(n, first, last)
            ab.append(pics)
            lineData.append(lineDataCurr)
            if (!last) {
                ab.append(Picture.hgap(hgap))
            }
            ldx += hgap + 2 * rad
        }
        val cb = canvasBounds
        picStack(
            linesPic(lineData),
            picRowCentered(ab),
            keyPic.withPosition(cb.x + 20, cb.y + 20)
        )
    }

    def linesPic(lineData: ArrayBuffer[ArrayBuffer[Point]]): Picture = {
        val totalLines = nDims.reduce(_ * _)
        if (totalLines > 5000) {
            println(s"Not showing connections ($totalLines)")
            Picture.circle(10).withNoPen()
        }
        else {
            val allPics = for (abPair <- lineData.sliding(2)) yield {
                val ab1 = abPair(0)
                val ab2 = abPair(1)
                val pics = for {
                    p1 <- ab1
                    p2 <- ab2
                } yield {
                    val len = math.sqrt(math.pow(p1.x - p2.x, 2) + math.pow(p1.y - p2.y, 2))
                    val angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
                    Picture.hline(len)
                        .withRotation(angle.toDegrees)
                        .withTranslation(p1.x, p1.y)
                        .withPenColor(black)
                        .withPenThickness(1)
                }
                picStack(pics)
            }
            picStack(allPics.toArray)
        }
    }

    def keyPic: Picture = {
        val input = picRowCentered(
            hiddenPicture(rad, true, false),
            Picture.hgap(20),
            Picture.text("Input")
                .withPenColor(darkGray)
        )
        val hidden = picRowCentered(
            hiddenPicture(rad, false, false),
            Picture.hgap(20),
            Picture.text("Hidden unit/neuron with a weight for each incoming connection, one bias, and relu activation")
                .withPenColor(darkGray)
        )
        val output = picRowCentered(
            hiddenPicture(rad, false, true),
            Picture.hgap(20),
            Picture.text("Output unit/neuron with a weight for each incoming connection, one bias, and softmax (layer) activation")
                .withPenColor(darkGray))
        picCol(output, Picture.vgap(5), hidden, Picture.vgap(5), input)
    }
}
