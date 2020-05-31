import sys
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import Row
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    conf = SparkConf().setAppName("Practica 4. Machine learning - celopez")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)

    # Leer los datos de los sets de entrenamiento y de test
    train_set = sqlContext.read.csv("/tmp/data/train_set.csv",header=True,sep=",",inferSchema=True)
    test_set = sqlContext.read.csv("/tmp/data/test_set.csv",header=True,sep=",",inferSchema=True)
    
    # Para que los dataframes sean aceptados por mllib hay que darles una formato concreto
    # Cambiamos en nombre de la columna objetivo y agrupamos las columnas de caracteristicas en una sola columna de arrays.
    train_set = train_set.withColumnRenamed("_c631","label")
    assembler = VectorAssembler(
    inputCols=['_c186', '_c245', '_c459', '_c221', '_c490', '_c429'],
    outputCol='features')
    train_set = assembler.transform(train_set)

    # Lo mismo para el conjunto de test
    test_set = test_set.withColumnRenamed("_c631","label")
    assembler = VectorAssembler(
    inputCols=['_c186', '_c245', '_c459', '_c221', '_c490', '_c429'],
    outputCol='features')
    test_set = assembler.transform(test_set)
    
    # Añadimos una columna con las caracteristicas escaladas entre 0 y 1
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    scalerModelTrain = scaler.fit(train_set)
    train_set = scalerModelTrain.transform(train_set)
    scalerModelTest = scaler.fit(test_set)
    test_set = scalerModelTrain.transform(test_set)


    ###### Entrenamiento de los modelos
    ### Regresion logistica 1
    # Entrenamiento
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="binomial")
    lrModel_1 = lr.fit(train_set)

    # Curva ROC
    roc = lrModel_1.summary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve: areaUnderROC: ' + str(lrModel_1.summary.areaUnderROC))
    plt.savefig('/tmp/data/ml_data/lrModel_1_ROC.png')

    # Valor de la curva ROC sobre el conjunto de testeo
    predictions = lrModel_1.transform(test_set)
    evaluator = BinaryClassificationEvaluator()
    f = open("/tmp/data/ml_data/lrModel_1_test_set_area_under_ROC.txt", "w+")
    f.write("Test set Area Under ROC: " + str(evaluator.evaluate(predictions)))

    # Valor de precision sobre el conjunto de testeo
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    f = open("/tmp/data/ml_data/lrModel_1_accuracy.txt", "w+")
    f.write("Test Accuracy: " + str(evaluator.evaluate(predictions)))


    ### Regresion logistica 2
    # Entrenamiento
    lr2 = LogisticRegression(maxIter=30, regParam=0.2, elasticNetParam=0.9, family="binomial")
    lrModel_2 = lr2.fit(train_set)

    # Curva ROC
    roc = lrModel_2.summary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve: areaUnderROC: ' + str(lrModel_2.summary.areaUnderROC))
    plt.savefig('/tmp/data/ml_data/lrModel_2_ROC.png')

    # Valor de la curva ROC sobre el conjunto de testeo
    predictions = lrModel_2.transform(test_set)
    evaluator = BinaryClassificationEvaluator()
    f = open("/tmp/data/ml_data/lrModel_2_test_set_area_under_ROC.txt", "w+")
    f.write("Test set Area Under ROC: " + str(evaluator.evaluate(predictions)))

    # Valor de precision sobre el conjunto de testeo
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    f = open("/tmp/data/ml_data/lrModel_2_accuracy.txt", "w+")
    f.write("Test Accuracy: " + str(evaluator.evaluate(predictions)))


    ### Naive Bayes 1
    # Entrenamos
    nb = NaiveBayes(smoothing=1.0, featuresCol="scaled_features", labelCol="label", modelType="multinomial")
    nbModel_1 = nb.fit(train_set)

    predictions = nbModel_1.transform(test_set)
    # Valor de precision sobre el conjunto de testeo
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    f = open("/tmp/data/ml_data/nbModel_1_accuracy.txt", "w+")
    f.write("Test Accuracy: " + str(evaluator.evaluate(predictions)))

    # Valor de la curva ROC sobre el conjunto de testeo
    evaluator = BinaryClassificationEvaluator()
    f = open("/tmp/data/ml_data/nbModel_1_test_set_area_under_ROC.txt", "w+")
    f.write("Test set Area Under ROC: " + str(evaluator.evaluate(predictions)))


    ### Naive Bayes 2
    # Entrenamos
    nb2 = NaiveBayes(smoothing=5, featuresCol="scaled_features", labelCol="label", modelType="multinomial")
    nbModel_2 = nb2.fit(train_set)

    predictions = nbModel_2.transform(test_set)
    # Valor de precision sobre el conjunto de testeo
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    f = open("/tmp/data/ml_data/nbModel_2_accuracy.txt", "w+")
    f.write("Test Accuracy: " + str(evaluator.evaluate(predictions)))

    # Valor de la curva ROC sobre el conjunto de testeo
    evaluator = BinaryClassificationEvaluator()
    f = open("/tmp/data/ml_data/nbModel_2_test_set_area_under_ROC.txt", "w+")
    f.write("Test set Area Under ROC: " + str(evaluator.evaluate(predictions)))


    ### Linear support vector machine  1
    # Entrenar
    lsvc = LinearSVC(maxIter=10, regParam=0.1)
    lsvcModel_1 = lsvc.fit(train_set)

    predictions = lsvcModel_1.transform(test_set)
    # Valor de precision sobre el conjunto de testeo
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    f = open("/tmp/data/ml_data/lsvcModel_1_accuracy.txt", "w+")
    f.write("Test Accuracy: " + str(evaluator.evaluate(predictions)))

    # Valor de la curva ROC sobre el conjunto de testeo
    evaluator = BinaryClassificationEvaluator()
    f = open("/tmp/data/ml_data/lsvcModel_1_test_set_area_under_ROC.txt", "w+")
    f.write("Test set Area Under ROC: " + str(evaluator.evaluate(predictions)))


    ### Linear support vector machine  2
    # Entrenar
    lsvc = LinearSVC(maxIter=50, regParam=0.02)
    lsvcModel_2 = lsvc.fit(train_set)

    predictions = lsvcModel_2.transform(test_set)
    # Valor de precision sobre el conjunto de testeo
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    f = open("/tmp/data/ml_data/lsvcModel_2_accuracy.txt", "w+")
    f.write("Test Accuracy: " + str(evaluator.evaluate(predictions)))

    # Valor de la curva ROC sobre el conjunto de testeo
    evaluator = BinaryClassificationEvaluator()
    f = open("/tmp/data/ml_data/lsvcModel_2_test_set_area_under_ROC.txt", "w+")
    f.write("Test set Area Under ROC: " + str(evaluator.evaluate(predictions)))


