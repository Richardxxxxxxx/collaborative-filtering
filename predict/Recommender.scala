package util


import model.AmazonRating
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.joda.time.{Seconds, DateTime}

import scala.util.Random

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.collection.mutable.ArrayBuffer  
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand





class Recommender(@transient sc: SparkContext, ratingFile: String) extends Serializable {
  val NumRecommendations = 10
  val MinRecommendationsPerUser = 10
  val MaxRecommendationsPerUser = 20
  val MyUsername = "myself"
  val NumPartitions = 20

  @transient val random = new Random() with Serializable
  // first create an RDD out of the rating file

  /*
  val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
  //val rawDate = hiveContext.sql("select * from "+table).rdd
  hiveContext.sql("show databases").show()
  hiveContext.sql("show tables").show()
  val rawDate = hiveContext.sql("select * from amz_rating").rdd
  val rawTrainingRatings = rawDate.map{case org.apache.spark.sql.Row(userId:String, productId:String, rating:Double,time:Int) =>AmazonRating(userId, productId, rating,"")}
  */
  val rawTrainingRatings = sc.textFile(ratingFile).map {
    line =>
      val Array(userId, productId, scoreStr) = line.split(",")
      AmazonRating(userId, productId, scoreStr.toDouble,"")
  }

  // only keep users that have rated between MinRecommendationsPerUser and MaxRecommendationsPerUser products
  val trainingRatings = rawTrainingRatings.groupBy(_.userId).filter(r => MinRecommendationsPerUser <= r._2.size  && r._2.size < MaxRecommendationsPerUser).flatMap(_._2).repartition(NumPartitions).cache()

  println(s"Parsed $ratingFile. Kept ${trainingRatings.count()} ratings out of ${rawTrainingRatings.count()}")

  // create user and item dictionaries
  val userDict = new Dictionary(MyUsername +: trainingRatings.map(_.userId).distinct.collect)
  val productDict = new Dictionary(trainingRatings.map(_.productId).distinct.collect)

  private def toSparkRating(amazonRating: AmazonRating) = {
    Rating(userDict.getIndex(amazonRating.userId),
      productDict.getIndex(amazonRating.productId),
      amazonRating.rating)
  }

  private def toAmazonRating(rating: Rating) = {
    AmazonRating(userDict.getWord(rating.user),
      productDict.getWord(rating.product),
      rating.rating,
      ""
    )
  }
  val sparkRatings = trainingRatings.map(toSparkRating)

  def getRandomProductId = productDict.getWord(random.nextInt(productDict.size))



  val model = MatrixFactorizationModel.load(sc, "ALS.model")
  val productFeatures = model.productFeatures//:(int,double[])
  val userFeatures = model.userFeatures//:(int,double[])
  var productLookUpTable = new ArrayBuffer[Int]()//n
  var pureProductFeatures = new ArrayBuffer[Double]()//n
  productFeatures.collect().foreach{case(product,feature)=>productLookUpTable+=product; pureProductFeatures++=feature}
  val featureDim = productFeatures.first()._2.length
  val productLen = productLookUpTable.length
  val productFeatureMatrix = new DenseMatrix(productLen,featureDim,pureProductFeatures.toArray) //Y n*f
  val YtY = inv(productFeatureMatrix.t * productFeatureMatrix)//f*f

  def predictOnly(ratings: Seq[AmazonRating]) = {


    //val ratings = Seq(toAmazonRating(new Rating(13,13,2)),toAmazonRating(new Rating(13,99,5)),toAmazonRating( new Rating(13,123,4)),toAmazonRating( new Rating(13,55,1)),toAmazonRating(new Rating(13,31,25)))

    // train model
    val myRatings = ratings.map(toSparkRating)
    //val myRatingRDD = sc.parallelize(myRatings)
    val userRating = DenseVector.zeros[Double](productLen)//n*1
    myRatings.foreach(rating=>userRating(productLookUpTable.indexOf(rating.product))=rating.rating)



    val startAls = DateTime.now
    val userFeature = userRating.t*productFeatureMatrix*YtY//1*f
    val predict = userFeature*productFeatureMatrix.t//1*n
    val pairs = (productLookUpTable zip predict.t.toArray).sortBy(-_._2).toArray

    // get ratings of all products not in my history ordered by rating (higher first) and only keep the first NumRecommendations
    val myUserId = userDict.getIndex(MyUsername)
    val endAls = DateTime.now
    val result = pairs.take(NumRecommendations).map{case(product,rating)=>toAmazonRating(new Rating(myUserId,product,rating))}
    val alsTime = Seconds.secondsBetween(startAls, endAls).getSeconds

    println(s"ALS Time: $alsTime seconds")
    result
  }

  def predict(ratings: Seq[AmazonRating]) = {
    // train model
    val myRatings = ratings.map(toSparkRating)
    val myRatingRDD = sc.parallelize(myRatings)

    val startAls = DateTime.now
    val model = ALS.train((sparkRatings ++ myRatingRDD).repartition(NumPartitions), 10, 20, 0.01)

    val myProducts = myRatings.map(_.product).toSet
    val candidates = sc.parallelize((0 until productDict.size).filterNot(myProducts.contains))

    // get ratings of all products not in my history ordered by rating (higher first) and only keep the first NumRecommendations
    val myUserId = userDict.getIndex(MyUsername)
    val recommendations = model.predict(candidates.map((myUserId, _))).collect
    val endAls = DateTime.now
    val result = recommendations.sortBy(-_.rating).take(NumRecommendations).map(toAmazonRating)
    val alsTime = Seconds.secondsBetween(startAls, endAls).getSeconds

    println(s"ALS Time: $alsTime seconds")
    result
  }

}