



import org.apache.spark.{ SparkContext, SparkConf }  
import org.apache.spark.SparkContext._ 
import org.apache.spark.mllib.recommendation.{ALS, Rating}

object recModel { 

	def main(args: Array[String]) { 
		if (args.length != 2) {
			System.err.println("Usage: tableName modelLocation")  
			System.exit(1)  
		}  
  
	    val conf = new SparkConf().setAppName("recModel")  
	    val sc = new SparkContext(conf)  

		class Dictionary(val words: Seq[String]) extends Serializable {
		  val wordToIndexMap = words.zipWithIndex.toMap

		  val getIndex = wordToIndexMap
		  val getWord = words

		  val size = words.size
		}

		case class AmazonRating(userId: String, productId: String, rating: Double)

		val NumRecommendations = 10
		val MinRecommendationsPerUser = 10
		val MaxRecommendationsPerUser = 20
		val MyUsername = "myself"
		val NumPartitions = 20
		/*
		val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
		val rawDate = hiveContext.sql("select * from "+args(0)).rdd
		val rawTrainingRatings = rawDate.map{
		    case org.apache.spark.sql.Row(userId:String, productId:String, score:Double,discard:Int) =>AmazonRating(userId, productId, score)
		}
		*/
		val rawTrainingRatings = sc.textFile(args(0)).map {line =>
      		val Array(userId, productId, scoreStr) = line.split(",")
      		AmazonRating(userId, productId, scoreStr.toDouble)
  		}
		
		val trainingRatings = rawTrainingRatings.groupBy(_.userId).filter(r => MinRecommendationsPerUser <= r._2.size  && r._2.size < MaxRecommendationsPerUser).flatMap(_._2).repartition(NumPartitions).cache()

		val userDict = new Dictionary(MyUsername +: trainingRatings.map(_.userId).distinct.collect)
		val productDict = new Dictionary(trainingRatings.map(_.productId).distinct.collect)

		def toSparkRating(amazonRating: AmazonRating) = {
		    Rating(userDict.getIndex(amazonRating.userId),
		    productDict.getIndex(amazonRating.productId),amazonRating.rating)
		}

		val sparkRatings = trainingRatings.map(toSparkRating)


		val model = ALS.train((sparkRatings).repartition(NumPartitions), 10, 20, 0.01)

		model.save(sc, args(1))
		sc.stop()  
		}
	}


