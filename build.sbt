name := "rec-model"

version := "1.0-SNAPSHOT"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.1"


libraryDependencies ++= Seq(
  // Spark
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
)

