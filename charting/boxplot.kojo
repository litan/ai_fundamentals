// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-marks-info.csv"))

val mathMarks = df.intColumn("Math")
val englishMarks = df.intColumn("English")
val box = boxPlot("Marks Info", "Math", mathMarks.asDoubleSeq)
addBoxPlotToChart(box, "English", englishMarks.asDoubleSeq)
drawChart(box)
