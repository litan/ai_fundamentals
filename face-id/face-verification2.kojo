// scroll down to the bottom of the file
// for code to play with

// #include /webcam.kojo
// #include /faceid.kojo

import net.kogics.kojo.util.Utils

cleari()
clearOutput()

setBackground(ColorMaker.hsl(0, 0.00, 0.06))

def centerPic(pic: Picture, w: Int, h: Int) {
    pic.translate(-w / 2, -h / 2)
}

val scriptDir = Utils.kojoCtx.baseDir

val faceDetector = new FaceDetector(s"$scriptDir/face_detection_model/")
val faceEmbedder = new FaceEmbedder(s"$scriptDir/face_feature_model")

var videoFramePic: Picture = _
var currImageMat: Mat = _
var currFaceRects: Seq[Rect] = _

runInBackground {
    val feed = new WebCamFeed()
    feed.startCapture { imageMat =>
        val faces = faceDetector.locateAndMarkFaces(imageMat)
        val vfPic2 = Picture.image(Java2DFrameUtils.toBufferedImage(imageMat))
        centerPic(vfPic2, imageMat.size(1), imageMat.size(0))
        vfPic2.draw()
        currImageMat = imageMat
        currFaceRects = faces
        if (videoFramePic != null) {
            videoFramePic.erase()
        }
        videoFramePic = vfPic2
    }
    println("feed done")
    scriptDone()
}

val btnWidth = 100
val gap = 10
val iRadius = 30

def button(label: String): Picture = {
    picStackCentered(
        Picture.rectangle(btnWidth, iRadius * 2).withFillColor(cm.lightBlue).withPenColor(gray),
        Picture.text(label).withPenColor(black)
    )
}

def scriptDone() {
    println("Closing Models...")
    faceDetector.close()
    faceEmbedder.close()
}

// -----------------------------
// Tweak stuff below as desired
// Some ideas:
// Tweak indicator colors
// Tweak threshold and explore false positives, false negatives, etc
// make buttons inactive while work is happening
// Learn an "average" face for better performance

val cb = canvasBounds
val learnButton = button("Learn")
val verifyButton = button("Verify")
val learnStatusIndicator = Picture.circle(iRadius).withPenColor(gray)
val verifyStatusIndicator = Picture.circle(iRadius).withPenColor(gray)

val panel =
    picRowCentered(
        learnStatusIndicator, Picture.hgap(gap),
        learnButton, Picture.hgap(gap), verifyButton,
        Picture.hgap(gap), verifyStatusIndicator
    )

draw(panel)
panel.setPosition(
    cb.x + (cb.width - panel.bounds.width) / 2 + iRadius,
    cb.y + iRadius + 5)

val similarityThreshold = 0.9
val fps = 5
var learnedEmbedding: Array[Float] = _

learnButton.onMouseClick { (x, y) =>
    if (currFaceRects.size == 1) {
        println("Learning Face")
        learnStatusIndicator.setFillColor(cm.purple)
        Utils.runAsyncQueued {
            val faceMat = faceEmbedder.extractAndResizeFace(currImageMat, currFaceRects(0))
            learnedEmbedding = faceEmbedder.faceEmbedding(faceMat)
            learnStatusIndicator.setFillColor(orange)
        }
    }
    else {
        println("There should be only one face on the screen for Learning")
        learnStatusIndicator.setFillColor(noColor)
        learnedEmbedding = null
    }

}

// normalized cosine similarity
def distance(a: Array[Float], b: Array[Float]): Float = {
    var ret = 0.0
    var mod1 = 0.0
    var mod2 = 0.0
    val length = a.length;
    for (i <- 0 until length) {
        ret += a(i) * b(i)
        mod1 += a(i) * a(i)
        mod2 += b(i) * b(i)
    }

    ((ret / math.sqrt(mod1) / math.sqrt(mod2) + 1) / 2.0).toFloat
}

def checkFace(imageMat: Mat, faces: Seq[Rect], learnedEmbedding: Array[Float]): Boolean = {
    if (faces.size == 1) {
        val faceMat = faceEmbedder.extractAndResizeFace(imageMat, faces(0))
        val emb = faceEmbedder.faceEmbedding(faceMat)
        val similarity = distance(emb, learnedEmbedding)
        println(s"Similarity - $similarity")
        if (similarity > similarityThreshold) true else false
    }
    else {
        println("Verification is done only if there is one face on the screen")
        false
    }
}

verifyButton.onMouseClick { (x, y) =>
    if (learnedEmbedding != null) {
        Utils.runAsyncQueued {
            val good = checkFace(currImageMat, currFaceRects, learnedEmbedding)
            if (good) {
                verifyStatusIndicator.setFillColor(green)
            }
            else {
                verifyStatusIndicator.setFillColor(red)
            }
            Utils.schedule(1) {
                verifyStatusIndicator.setFillColor(noColor)
            }
        }
    }
    else {
        println("First Learn, then Verify!")
    }
}