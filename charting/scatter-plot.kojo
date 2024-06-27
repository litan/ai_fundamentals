// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-marks-info.csv"))

val mathMarks = df.intColumn("Math")
val englishMarks = df.intColumn("English")
val scatter = scatterChart("Correlation between Math and English Marks", "Math", "English", mathMarks.asDoubleSeq, englishMarks.asDoubleSeq)
drawChart(scatter)

println(mathMarks.mean)
println(mathMarks.median)
println(mathMarks.standardDeviation)
println(mathMarks.variance)
println(mathMarks.correlation(englishMarks))

