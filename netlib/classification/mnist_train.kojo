// #include /nn.kojo
// #include /plot.kojo
// #include classification-net

import ai.djl.basicdataset.cv.classification.Mnist

val seed = 40
initRandomGenerator(seed)
Engine.getInstance.setRandomSeed(seed)

cleari()
clearOutput()

def dataset(usage: Dataset.Usage, mgr: NDManager) = {
    val mnist =
        Mnist.builder()
            .optUsage(usage)
            .setSampling(64, true)
            .optManager(mgr)
            .build()

    mnist.prepare(new ProgressBar())
    mnist
}

def learningRate(e: Int) = e match {
    case n if n <= 10 => 0.1f
    case n if n <= 15 => 0.03f
    case _            => 0.01f
}

ndScoped { use =>
    val model = use(new ClassificationNet(784, 38, 12, 10))
    val trainingSet = dataset(Dataset.Usage.TRAIN, model.nm)
    val valSet = dataset(Dataset.Usage.TEST, model.nm)
    val numEpochs = 20
    val lossChart = new LiveChart(
        "Loss Plot", "epoch", "loss", 0, numEpochs, 0, 1
    )

    val tl = new TrainingListener {
        def onEpochDone(epoch: Int, loss: Double) {
            lossChart.update(epoch, loss)
        }
    }

    model.setTrainingListener(tl)

    timeit("Training") {
        model.train(trainingSet, valSet, numEpochs, learningRate)
    }
    model.showAccuracy(valSet)
    model.save()
}
