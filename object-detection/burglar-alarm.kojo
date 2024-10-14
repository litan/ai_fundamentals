// scroll down to the bottom of the file
// for code to play with

// #include /webcam.kojo
// #include /object-detection.kojo

import net.kogics.kojo.util.Utils

cleari()
clearOutput()

setBackground(ColorMaker.hsl(0, 0.00, 0.06))

def centerPic(pic: Picture, w: Int, h: Int) {
    pic.translate(-w / 2, -h / 2)
}

val scriptDir = Utils.kojoCtx.baseDir
val objectDetector = new ObjectDetector(s"$scriptDir/yolov8n/")

var currFramePic: Picture = _
var currImageMat: Mat = _
var currObjects: DetectedObjects = _

runInBackground {
    // feed from device 0 (default monitor) at 5 fps
    val feed = new WebCamFeed(0, 5)
    feed.startCapture { imageMat =>
        val (detections, markedImage) = objectDetector.findObjects(imageMat)
        val nextFramePic = Picture.image(markedImage)
        centerPic(nextFramePic, imageMat.size(1), imageMat.size(0))
        nextFramePic.draw()
        currImageMat = imageMat
        currObjects = detections
        if (currFramePic != null) {
            currFramePic.erase()
        }
        currFramePic = nextFramePic
        updateBurglarIndicator(detections)
    }
    println("feed done")
    scriptDone()
}

def scriptDone() {
    println("Closing Models...")
    objectDetector.close()
}

val cb = canvasBounds
val iRadius = 30
val burglarStatusIndicator = Picture.circle(iRadius).withPenColor(gray)
burglarStatusIndicator.setFillColor(orange)

val panel =
    picRowCentered(
        burglarStatusIndicator
    )

draw(panel)
panel.setPosition(
    cb.x + (cb.width - panel.bounds.width) / 2 + iRadius,
    cb.y + iRadius + 10)

def updateBurglarIndicator(detections: DetectedObjects) {
    import scala.jdk.CollectionConverters._
    val detectedClasses = detections.getClassNames.asScala
    if (detectedClasses.contains("person")) {
        burglarStatusIndicator.setFillColor(red)
    }
    else {
        burglarStatusIndicator.setFillColor(green)
    }
}

