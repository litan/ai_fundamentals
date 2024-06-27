// #include /dataframe.kojo

cleari()
clearOutput()

val df = readCsv(resolvedPath("student-marks-info.csv"))

val mathMarks = df.intColumn("Math")
val name = df.stringColumn("Name")
val area = df.stringColumn("Area")
val interestArea = "Vasant Vihar"

val resultDf = df.select(name, mathMarks).where(area.isEqualTo(interestArea).and(mathMarks.isGreaterThan(50)))

val bar = barChart(
    s"High schoring Math students in $interestArea",
    "Name",
    "Marks",
    resultDf.stringColumn("Name").asStringSeq,
    resultDf.intColumn("Math").asIntSeq
)
drawChart(bar)
