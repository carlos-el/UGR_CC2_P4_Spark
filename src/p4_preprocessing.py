import sys
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.functions import col, explode, array, lit

if __name__ == "__main__":
    conf = SparkConf().setAppName("Practica 4. Preprocessing - celopez")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    # Leer los datos de las columnas a usar 
    df = sqlContext.read.csv("/tmp/data/part-000.csv",header=True,sep=",",inferSchema=False)
    # Obtener los datos del summary como una tabla
    df.describe().repartition(1).write.option("header","true").csv("/tmp/data/preprocessing_data/summary_df")
    

    # Identificar el ratio de desbalanceo del dataset segun la varaible objetivo (_c631)
    major_df = df.filter(col("_c631") == 0)
    minor_df = df.filter(col("_c631") == 1)
    ratio = float(major_df.count()/minor_df.count())
    f = open("/tmp/data/preprocessing_data/unbalanced_ratio.txt", "w+")
    f.write("Ratio de desbalanceo: " + str(ratio))

    # Hacer un subsamplig de la clase con mas presencia en el dataset
    sub_major_df = major_df.sample(False, 1/ratio)
    undersampled_df = sub_major_df.unionAll(minor_df)
    # Obtener los datos del summary como una tabla
    undersampled_df.describe().repartition(1).write.option("header","true").csv("/tmp/data/preprocessing_data/summary_undersampled_df")

    # Dividir en train y test 
    splits = undersampled_df.randomSplit([0.2, 0.8])
    test_set = splits[0]
    train_set = splits[1]

    # Obtener los datos del summary como una tabla
    train_set.describe().repartition(1).write.option("header","true").csv("/tmp/data/preprocessing_data/summary_train_set")
    test_set.describe().repartition(1).write.option("header","true").csv("/tmp/data/preprocessing_data/summary_test_set")

    train_set.repartition(1).write.option("header","true").csv("/tmp/data/preprocessing_data/train_set")
    test_set.repartition(1).write.option("header","true").csv("/tmp/data/preprocessing_data/test_set")

    sc.stop()