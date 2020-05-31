import sys
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window

if __name__ == "__main__":
    conf = SparkConf().setAppName("Practica 4. Get columns - celopez")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    ### Primero vamos a leer la cabecara para poder sacar el numero de columna de nuestras columnas en el fichero .data
    # Especificar las columnas escogidas (la variable de clase la primera)
    col_names = ["class", "PSSM_r1_-4_S", "PSSM_r1_-1_P", "PSSM_r2_1_H", "PSSM_r1_-2_L", "PSSM_r2_2_V", "PSSM_r2_-1_Y"]
    # Leer la cabacera como dataframe de una sola columna (nombrada _c0 por defecto) 
    df_header = sqlContext.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.header",header=False,sep=",",inferSchema=False)
    # Aniadimos un campo que indique el numero de columna que corresponde a cada atributo en el fichero .data 
    df_header = df_header.withColumn('row_num', row_number().over(Window.orderBy(monotonically_increasing_id())) - 2)
    df_header.show()

    # Permitimos consultas sql en el dataframe
    df_header.createOrReplaceTempView("sql_header")
    col_query = ""
    # Por cada columna buscamos su numero de fila y lo aniadimos a un string precedido de '_c' 
    # (spark al cargar un df si no hay header da a las columnas los nombre _c0, _c1, etc)
    for idx, col in enumerate(col_names):
        sql_row_num = sqlContext.sql("SELECT row_num FROM sql_header where _c0 like '@attribute "+ col +"%'").collect()[0]["row_num"]
        if idx == 0:
            col_query += "_c" + str(sql_row_num)
        else:
            col_query += ", _c" + str(sql_row_num)
    # Asi creamos el select de la query que usaremos para extrae las columnas requeridas del archivo .data
    print(col_query)

    ### Extraemos las columnas del fichero de datos y las guardamos en un solo fichero en hdfs.
    df = sqlContext.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.data",header=False,sep=",",inferSchema=False)
    df.createOrReplaceTempView("sql_dataset")
    sqlDF = sqlContext.sql("SELECT " + col_query + " FROM sql_dataset")  
    sqlDF.show()
    sqlDF.repartition(1).write.option("header","true").csv("p4_columns")

    sc.stop()