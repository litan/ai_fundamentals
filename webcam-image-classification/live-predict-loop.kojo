// #include /webcam.kojo
// #include live-predict.kojo

import net.kogics.kojo.util.Utils
import scala.collection.{ JavaConverters => JC }

cleari()
clearOutput()

val cb = canvasBounds

def centerPic(pic: Picture, w: Int, h: Int) {
    pic.translate(-w / 2, -h / 2)
}

val scriptDir = Utils.kojoCtx.baseDir

var currFramePic: Picture = _
var currPredPic: Picture = _

val liveArea = Picture.rectangle(224, 224)
    .withPenColor(cyan).withTranslation(-112, -112).withPenThickness(4)

liveArea.draw()

runInBackground {
    // feed from device 0 (default monitor) at 10 fps
    val feed = new WebCamFeed(0, 10)
    feed.startCapture { imageMat =>
        val img = matToBufferedImage(imageMat)
        val nextFramePic = Picture.image(img)
        centerPic(nextFramePic, imageMat.size(1), imageMat.size(0))
        nextFramePic.draw()
        liveArea.moveToFront()
        if (currFramePic != null) {
            currFramePic.erase()
        }
        currFramePic = nextFramePic
        val pred = predict(ImageFactory.getInstance.fromImage(img))
        showPredictions(JC.asScala(pred.getClassNames).toSeq, JC.asScala(pred.getProbabilities).toSeq)
    }
    println("feed done")
    scriptDone()
}

val font = Font("Serif", 21)
def showPredictions(classNames: Seq[String], probs: Seq[java.lang.Double]) {
    val pics = ArrayBuffer.empty[Picture]
    classNames.zip(probs).foreach { case (cls, prob) =>
        val pic = picRowCentered(
            picStackCentered(
                Picture.rectangle(100, 20).withNoPen(),
                Picture.text(cls, font, black),
            ),
            Picture.hgap(10),
            picStack(
                Picture.rectangle(100, 20).withPenColor(cm.gray).withPenThickness(1),
                Picture.rectangle(100 * prob, 20).withFillColor(cm.darkOrange).withNoPen()
            )
        )
        pics.append(pic)
        pics.append(Picture.vgap(5))
    }
    val nextPredPic = picColCentered(pics)
    nextPredPic.moveToBack()
    nextPredPic.draw()
    val b = nextPredPic.bounds
    nextPredPic.setPosition(-b.width, cb.y)
    
    if (currPredPic != null) {
        currPredPic.erase()
    }
    currPredPic = nextPredPic
}

def scriptDone() {
    println("Closing Models...")
    closePredictor()
}

