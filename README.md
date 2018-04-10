# collaborative-filtering


/opt/spark-2.2.1-bin-hadoop2.7/bin/spark-submit \
--master yarn \
--deploy-mode client \
rec-model_2.11-1.0-SNAPSHOT.jar \
xxxxxx /yyyyyyy


xxxx refers to the data to be trained

/yyyyyyy referes to the model path to be stored


collaborative-filtering/predict/fold-in.scala is part of fold-in implementation

![alt text](https://github.com/Richardxxxxxxx/collaborative-filtering/blob/master/image/fold-in.png)


collaborative-filtering/predict/fold-in.scala implement two ways of predition for a new user

def predictOnly(ratings: Seq[AmazonRating]) using fold-in idea

def predict(ratings: Seq[AmazonRating]) training the model for all users including the new user to get the 
new user's feature. After getting the new user's feature, using it to make the prediction.
