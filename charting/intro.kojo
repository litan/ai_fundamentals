// #include /plot.kojo
// #include /dataframe.kojo

cleari()
clearOutput()

// Load a couple of dataframes from csv files in the kojo-ai data dir
val marks = readCsv(resolvedPath("student-marks.csv"))
val info = readCsv(resolvedPath("student-info.csv"))

// Do an inner join on the frames based on the "Name" field
val marksInfo = marks.joinOn("Name").inner(info)

// Find mean math marks grouped by area
marksInfo.summarize("Math", AggregateFunctions.mean).by("Area")

// Do a selection - find all students with math marks greater than 90
marksInfo.where(marksInfo.intColumn("Math").isGreaterThan(90))

// draw a histogram (with 7 bins) of math marks
//drawChart(marksInfo.columns(Seq("Math")).makeHistogram(7))

// draw a bar chart of the areas where the students live
 drawChart(marksInfo.columns(Seq("Area")).makeBarChart())

// drawChart(marksInfo.columns(Seq("Area")).makePieChart())

val mathMarks = marksInfo.numberColumn("Math")
val engMarks = marksInfo.numberColumn("English")
val area = marksInfo.stringColumn("Area")
marksInfo.where(mathMarks.isGreaterThan(80)).summarize(mathMarks, AggregateFunctions.mean).by(area)
marksInfo.select(mathMarks, engMarks).where(mathMarks.isGreaterThan(80))