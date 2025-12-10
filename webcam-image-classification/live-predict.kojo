import java.nio.file.{ Files, Paths, Path }

import ai.djl._
import ai.djl.engine._
import ai.djl.modality._
import ai.djl.modality.cv._
import ai.djl.modality.cv.transform._
import ai.djl.modality.Classifications
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

import ai.djl.inference.Predictor
import ai.djl.modality.cv.translator.ImageClassificationTranslator
import scala.jdk.CollectionConverters._
import ai.djl.modality.Classifications.Classification

val imageSize = 224

// Where we saved the model in trainThumbsModel
val modelDir = Paths.get(s"${kojoCtx.baseDir}/model")
val modelName = "resnet18-transfer"

// We'll lazy-load and cache these:
var model: Model = _
var predictor: Predictor[Image, Classifications] = _

// -----------------------------------------
// Helpers
// -----------------------------------------

def parseClassesProperty(model: Model): java.util.List[String] = {
    val raw = Option(model.getProperty("Classes")).getOrElse("[]")
    val trimmed = raw.trim.stripPrefix("[").stripSuffix("]")
    val names =
        if (trimmed.isEmpty) Nil
        else trimmed.split(",").toList.map(_.trim)
    names.asJava
}

import ai.djl.modality.Classifications
import ai.djl.modality.Classifications.Classification

var resnet18EmbeddingModel: ZooModel[NDList, NDList] = _
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

def numClasses = {
    import java.io.File
    val dataDir = new File(s"${kojoCtx.baseDir}/data")
    Option(dataDir.listFiles) // handle null safely
        .getOrElse(Array.empty[File])
        .count(_.isDirectory)
}

def initPredictor(): Unit = {
    if (predictor != null) return

    val model0 = makeModelPretrainedResNet(numClasses)
    model0.load(modelDir, modelName)

    val synset = parseClassesProperty(model0)

    val translator =
        ImageClassificationTranslator
            .builder()
            .addTransform(new CenterCrop(imageSize, imageSize))
            .addTransform(new ToTensor())
            .addTransform(
                new Normalize(
                    Array(0.485f, 0.456f, 0.406f),
                    Array(0.229f, 0.224f, 0.225f)
                )
            )
            .optApplySoftmax(true)
            .optSynset(synset)
            .build()

    model = model0
    predictor = model.newPredictor(translator)
}

def closePredictor(): Unit = {
    if (predictor != null) {
        predictor.close()
        predictor = null
    }
    if (model != null) {
        model.close()
        model = null
    }
}

def predict(image: Image): Classifications = {
    if (predictor == null) {
        initPredictor()
    }
    predictor.predict(image)
}

// -----------------------------------------
// Simple file-based test
// -----------------------------------------

//import ai.djl.modality.cv.ImageFactory
//
//val testPath = Paths.get("/home/lalit/work/ai_fundamentals/transfer-learning/data/thumbsdown/106.jpg")
//
//// Load directly as a DJL Image from file
//val djlImg: Image = ImageFactory.getInstance().fromFile(testPath)
//
//val label = predictThumb(djlImg)
//println(s"Prediction: $label")
