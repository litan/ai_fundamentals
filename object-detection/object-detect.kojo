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
val objectDetector = new ObjectDetector(s"$scriptDir/model/")

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
    }
    println("feed done")
    scriptDone()
}

def scriptDone() {
    println("Closing Models...")
    objectDetector.close()
}

