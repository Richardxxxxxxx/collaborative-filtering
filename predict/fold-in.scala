
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.collection.mutable.ArrayBuffer  
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand

val model = MatrixFactorizationModel.load(sc, "/testmodel")

val productFeatures = model.productFeatures
val userFeatures = model.userFeatures



var productLookUpTable = new ArrayBuffer[Int]()//n
var pureProductFeatures = new ArrayBuffer[Double]()//n

productFeatures.collect().foreach{case(product,feature)=>productLookUpTable+=product; pureProductFeatures++=feature}

val featureDim = productFeatures.first()._2.length
val productLen = productLookUpTable.length

val productFeatureMatrix = new DenseMatrix(productLen,featureDim,pureProductFeatures.toArray) //Y n*f

val YtY = inv(productFeatureMatrix.t * productFeatureMatrix)//f*f


val userRating = DenseVector.zeros[Double](productLen)//n*1
//userRating(1,4) = 1

val userFeature = userRating.t*productFeatureMatrix*YtY//1*f

val predict = userFeature*productFeatureMatrix.t//1*n

val pairs = (productLookUpTable zip predict.t.toArray).sortBy(-_._2).toArray