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
        val xValues = xNormalizer.transform(xValuesRaw).map(_.toFloat)
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

    val rad = 20

    def inputPicture(r: Int) = Picture.circle(r).withFillColor(cm.gray).withNoPen()
    def hiddenPicture(r: Int, last: Boolean) = {
        val fillc = if (last) cm.blue else cm.green
        Picture.circle(r).withFillColor(fillc).withNoPen()
    }

    def visualize() {
        val vgap = 20
        val hgap = 60
        var ldx = 0
        var ldy = 0

        val lineData = ArrayBuffer.empty[ArrayBuffer[Point]]
        var lineDataCurr = ArrayBuffer.empty[Point]

        def vertPics(n: Int, last: Boolean): Picture = {
            ldy = rad - (n * 2 * rad + (n - 1) * vgap) / 2
            val ab = ArrayBuffer.empty[Picture]
            repeatFor(1 to n) { idx =>
                val pic = hiddenPicture(rad, last)
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
        val inputPic = inputPicture(rad)
        ab.append(inputPic)
        ab.append(Picture.hgap(hgap))
        lineDataCurr.append(Point(ldx, ldy))
        lineData.append(lineDataCurr)
        val gparams = params.grouped(2).toList
        for ((ppair, cnt) <- gparams.zipWithIndex) {
            val last = (cnt == gparams.length - 1)
            val b = ppair(1)
            val n = b.getShape.get(0).toInt
            ldx += hgap + 2 * rad
            lineDataCurr = ArrayBuffer.empty[Point]
            val pics = vertPics(n, last)
            ab.append(pics)
            lineData.append(lineDataCurr)
            if (!last) {
                ab.append(Picture.hgap(hgap))
            }
        }
        val cb = canvasBounds
        val netPic = picStack(
            linesPic(lineData),
            picRowCentered(ab),
            keyPic.withPosition(cb.x + 20, cb.y + 20)
        )
        draw(netPic)
    }

    def linesPic(lineData: ArrayBuffer[ArrayBuffer[Point]]): Picture = {
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

    def keyPic: Picture = {
        val input = picRowCentered(
            inputPicture(rad),
            Picture.hgap(20),
            Picture.text("Input")
                .withPenColor(darkGray)
        )
        val hidden = picRowCentered(
            hiddenPicture(rad, false),
            Picture.hgap(20),
            Picture.text("Hidden unit/neuron with a weight for each incoming connection, one bias, and an activation function")
                .withPenColor(darkGray)
        )
        val output = picRowCentered(
            hiddenPicture(rad, true),
            Picture.hgap(20),
            Picture.text("Output unit/neuron with a weight for each incoming connection, and one bias")
                .withPenColor(darkGray))
        picCol(output, Picture.vgap(5), hidden, Picture.vgap(5), input)
    }
}
