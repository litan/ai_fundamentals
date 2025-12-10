// #include /webcam.kojo

import javax.imageio.ImageIO
import java.io.File

cleari()
clearOutput()
disablePanAndZoom()
val cb = canvasBounds

//setBackground(ColorMaker.hsl(0, 0.00, 0.06))

def centerPic(pic: Picture, w: Int, h: Int) {
    pic.translate(-w / 2, -h / 2)
}

val scriptDir = kojoCtx.baseDir
var currFramePic: Picture = _
val label = "classlabel"
var ctr = 1

val fl = new File(s"${kojoCtx.baseDir}/datax/$label/")
fl.mkdirs()
println(s"Data directory is: $fl")

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
        saveImage(img)
    }
    println(s"Stopped")
}

def saveImage(img: java.awt.image.BufferedImage) {
    if (started) {
        val file = new File(s"${kojoCtx.baseDir}/datax/$label/$ctr.jpg")
        ImageIO.write(img, "jpg", file)
        println(ctr)
        ctr += 1
    }
}

var started = false

val font = Font("Dialog", 20, BoldFont)
val startBtnRect = Picture.rectangle(100, 50)
startBtnRect.setPenColor(cm.green)
startBtnRect.setFillColor(cm.green)
val startBtn = picStackCentered(startBtnRect, Picture.text("Start", font, white))
startBtn.onMouseClick { (x, y) =>
    started = true
}

val stopBtnRect = Picture.rectangle(100, 50)
stopBtnRect.setFillColor(red)
val stopBtn = picStackCentered(stopBtnRect, Picture.text("Stop", font, white))
stopBtn.onMouseClick { (x, y) =>
    started = false
}

val row = picRow(startBtn, Picture.hgap(10), stopBtn)
draw(row)
val b = row.bounds
row.setPosition(-b.width / 2, cb.y + 5)
