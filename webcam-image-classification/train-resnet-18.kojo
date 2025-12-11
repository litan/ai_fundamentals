// -----------------------------------------------
// Imports
// -----------------------------------------------
import java.nio.file.{ Files, Paths, Path }

import ai.djl._
import ai.djl.engine._
import ai.djl.modality._
import ai.djl.modality.cv._
import ai.djl.modality.cv.transform._
import ai.djl.ndarray.types._
import ai.djl.ndarray.{ NDArray, NDList }
import ai.djl.nn._
import ai.djl.nn.core.Linear
import ai.djl.training._
import ai.djl.training.dataset._
import ai.djl.training.evaluator._
import ai.djl.training.listener._
import ai.djl.training.loss._
import ai.djl.training.optimizer._
import ai.djl.training.tracker._
import ai.djl.training.util._
import ai.djl.translate._
import ai.djl.basicdataset.cv.classification.ImageFolder
import ai.djl.basicmodelzoo.cv.classification.ResNetV1
import ai.djl.repository.zoo.{ Criteria, ZooModel }
import ai.djl.metric.Metrics
import ai.djl.pytorch.jni.JniUtils

Engine.getInstance
JniUtils.setGradMode(false)
JniUtils.setGraphExecutorOptimize(false)
net.kogics.kojo.nn.resetGradientCollection()

// -----------------------------------------------
// Config
// -----------------------------------------------

// For ResNet-18 embedding (pretrained on ImageNet)
val imageSize = 224
val batchSize = 16
val numEpochs = 2 // bump up/down depending on speed
val learningRate = 3e-3f

// Keep a strong reference to the embedding ZooModel so its native
// resources are NOT freed while we're training with its block.
var resnet18EmbeddingModel: ZooModel[NDList, NDList] = _

// -----------------------------------------------
// Dataset: load dataset via ImageFolder
// -----------------------------------------------

def makeDataset(datasetRoot: String): ImageFolder = {
    val dataset =
        ImageFolder
            .builder()
            .setRepositoryPath(Paths.get(datasetRoot))
            .optMaxDepth(1)
            .addTransform(new CenterCrop(imageSize, imageSize))
            .addTransform(new ToTensor())
            .addTransform(
                new Normalize(
                    Array(0.485f, 0.456f, 0.406f), // mean
                    Array(0.229f, 0.224f, 0.225f) // std
                )
            )
            .setSampling(batchSize, true) // shuffle between epochs
            .build()

    dataset.prepare(new ProgressBar())
    dataset
}

// -----------------------------------------------
// Model: Pretrained ResNet-18 embedding + small classifier head
// -----------------------------------------------

def makeModelPretrainedResNet(numClasses: Int): Model = {
    // 1. Load (or reuse) pretrained ResNet-18 embedding from PyTorch zoo
    if (resnet18EmbeddingModel == null) {
        val criteria = Criteria
            .builder()
            .setTypes(classOf[NDList], classOf[NDList])
            .optModelUrls("djl://ai.djl.pytorch/resnet18_embedding")
            .optEngine("PyTorch")
            .optProgress(new ProgressBar())
            // tell DJL not to train the embedding params
            .optOption("trainParam", "false")
            .build()

        resnet18EmbeddingModel = criteria.loadModel()
    }

    val baseBlock: Block = resnet18EmbeddingModel.getBlock

    // Extra safety: freeze all parameters in the pretrained backbone
    baseBlock.freezeParameters(true)

    // 2. Build transfer-learning block:
    //    [resnet18 embedding] -> [squeeze H,W] -> [Linear(numClasses)]
    val transferBlock = new SequentialBlock()
    transferBlock.add(baseBlock)
    // embedding output shape is (batch, C, 1, 1); squeeze spatial dims -> (batch, C)
    transferBlock.addSingleton((nd: NDArray) => nd.squeeze(Array(2, 3)))
    transferBlock.add(
        Linear
            .builder()
            .setUnits(numClasses.toLong)
            .build()
    )

    // 3. Wrap in a Model
    val model = Model.newInstance("resnet18-transfer")
    model.setBlock(transferBlock)

    model
}

// -----------------------------------------------
// Training config
// -----------------------------------------------

def makeTrainingConfig(): DefaultTrainingConfig = {
    val loss = Loss.softmaxCrossEntropyLoss()
    val lrTracker = Tracker.fixed(learningRate)
    val optimizer = Optimizer.adam()
        .optLearningRateTracker(lrTracker)
        .build()

    new DefaultTrainingConfig(loss)
        .optOptimizer(optimizer)
        // Be explicit: we know we're using PyTorch
        .optDevices(Engine.getEngine("PyTorch").getDevices(1))
        .addEvaluator(new Accuracy()) // track accuracy
        .addTrainingListeners(TrainingListener.Defaults.logging(): _*)
    //        .addTrainingListeners(
    //            new BatchMetricsPrinter,
    //            new EpochMetricsPrinter
    //        )
}

// -----------------------------------------------
// Train and save the model (with train/val split)
// -----------------------------------------------

def trainModel(
    datasetRoot:    String,
    modelOutputDir: String
): Unit = {

    // 0. clean out old model
    val outDir: Path = Paths.get(modelOutputDir)
    Files.createDirectories(outDir)
    val iter = Files.list(outDir).iterator
    while (iter.hasNext) {
        Files.delete(iter.next)
    }

    // 1. Full dataset
    val fullDataset = makeDataset(datasetRoot)
    val synset = fullDataset.getSynset // class names inferred from folder names
    val numClasses = synset.size()

    println(s"Found $numClasses classes: " + synset)

    // 1b. Split into train / validation (e.g. 80% / 20%)
    val Array(trainingSet, validateSet) =
        fullDataset.randomSplit(8, 2) // ratios 8:2 → 80% train, 20% val

    // 2. Model + trainer (using pretrained ResNet-18 transfer model)
    val model = makeModelPretrainedResNet(numClasses)
    val config = makeTrainingConfig()
    val trainer = model.newTrainer(config)

    // Enable metrics collection (for listener logging)
    trainer.setMetrics(new Metrics())

    val inputShape = new Shape(1, 3, imageSize, imageSize)
    trainer.initialize(inputShape)

    // 3. Training + validation loop (DJL handles both)
    EasyTrain.fit(trainer, numEpochs, trainingSet, validateSet)

    // 4. Save the model
    model.setProperty("Epoch", numEpochs.toString)
    model.setProperty("ImageSize", imageSize.toString)
    model.setProperty("Classes", synset.toString)

    model.save(outDir, "resnet18-transfer")

    println(s"Model saved to: $outDir")

    val result = trainer.getTrainingResult

    val finalTrainLoss = result.getTrainLoss
    val finalValLoss = result.getValidateLoss
    val finalTrainAcc = result.getTrainEvaluation("Accuracy")
    val finalValAcc = result.getValidateEvaluation("Accuracy")

    println(f"Final metrics -- trainAcc=$finalTrainAcc%.3f, valAcc=$finalValAcc%.3f, " +
        f"trainLoss=$finalTrainLoss%.3f, valLoss=$finalValLoss%.3f")

    // 5. Cleanup
    trainer.close()
    model.close()
    if (resnet18EmbeddingModel != null) {
        resnet18EmbeddingModel.close()
        resnet18EmbeddingModel = null
    }
}

def showStatus(msg: String) {
    erasePictures()
    drawCentered(
        Picture.text(msg, 40, black)
    )
}

cleari()
clearOutput()
showStatus("Model training in progress...")
val dataRoot = s"${kojoCtx.baseDir}/data"
val modelDir = s"${kojoCtx.baseDir}/model"
trainModel(dataRoot, modelDir)
showStatus("Model training done.")