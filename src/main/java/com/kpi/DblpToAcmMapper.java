package com.kpi;

import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import static com.kpi.LabHelper.*;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

public class DblpToAcmMapper {
    public static void main(String[] args) {
        final SparkSession sparkSession = SparkSession.builder()
                .appName(APP_NAME)
                .master(MASTER)
                .config("spark.sql.warehouse.dir", "src/main/resources/")
                .getOrCreate();

        final DataFrameReader dataFrameReader = sparkSession.read();
        dataFrameReader.option("header", "true");
        final Dataset<Row> acmDataSet = dataFrameReader.csv(ACM_CSV).filter("year < 2000");
        final Dataset<Row> dblp2DataSet = dataFrameReader.csv(DBLP2_CSV).filter("year < 2000");

        Tokenizer tokenizer = new Tokenizer().setInputCol("title").setOutputCol("words");
        HashingTF hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures");

        IDF idf = new IDF()
                .setInputCol("rawFeatures")
                .setOutputCol("features");

        Dataset<Row> acmWordsData = tokenizer.transform(acmDataSet);
        Dataset<Row> hashingAcmDataSet = hashingTF.transform(acmWordsData);
        IDFModel idfAcmModel = idf.fit(hashingAcmDataSet);
        Dataset<Row> acmDataset1 = idfAcmModel.transform(hashingAcmDataSet).select("id", "title", "features")
                .withColumnRenamed("id", "idACM");


        Dataset<Row> dblp2WordsData = tokenizer.transform(dblp2DataSet);
        Dataset<Row> hashingDblp2DataSet = hashingTF.transform(dblp2WordsData);
        IDFModel idfDblp2Model = idf.fit(hashingDblp2DataSet);
        Dataset<Row> dblp2Dataset1 = idfDblp2Model.transform(hashingDblp2DataSet).select("id", "title", "features")
                .withColumnRenamed("id", "idDBLP2")
                .withColumnRenamed("features", "dblp2_features");

        Dataset<Row> merged = acmDataset1.crossJoin(dblp2Dataset1);
        sparkSession.udf().register(
                "distance", (Vector f1, Vector f2) -> cosSim(f1.toArray(), f2.toArray()), DataTypes.DoubleType);

        merged.withColumn("dist", callUDF("distance", col("features"), col("dblp2_features")))
                .select("idDBLP2", "idACM", "dist")
                .filter("dist > 0.5")
                .show(false);
    }

    private static Double cosSim(double[] vector1, double[] vector2) {
        double result1 = 0d;
        double result2 = 0d;
        double result3 = 0d;

        for (int i = 0; i < vector1.length; i++) {
            result1 += vector1[i] * vector2[i];
        }

        for (int i = 0; i < vector1.length; i++) {
            result2 += vector1[i] * vector1[i];
            result3 += vector2[i] * vector2[i];
        }

        return result1 / (Math.sqrt(result2) * Math.sqrt(result3));

    }
}
