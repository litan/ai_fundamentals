// #include /plot.kojo
// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-marks-info.csv"))

val mathMarks = df.intColumn("Math")
val englishMarks = df.intColumn("English")
val hist = histogram("Math Marks and more...", "Marks", "# Students", mathMarks.asDoubleSeq, 4)
addHistogramToChart(hist, Some("English"), englishMarks.asDoubleSeq, 4)
drawChart(hist)