def range(start: Int, end: Int, step: Double = 1.0): Array[Double] = {
    rangeTill(start, end, step).map(_.toDouble).toArray
}
