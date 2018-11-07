package com.kpi;

import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static com.kpi.LabHelper.*;

public class DblpToAcmMapper {
    public static void main(String[] args) {
        final SparkSession sparkSession = SparkSession.builder()
                .appName(APP_NAME)
                .master(MASTER)
                .config("spark.sql.warehouse.dir", "src/main/resources/")
                .getOrCreate();

        final DataFrameReader dataFrameReader = sparkSession.read();
        dataFrameReader.option("header", "true");
        final Dataset<Row> acmDataSet = dataFrameReader.csv(ACM_CSV);
        final Dataset<Row> dblp2DataSet = dataFrameReader.csv(DBLP2_CSV);

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
        Dataset<Row> acmDataset1 = idfAcmModel.transform(hashingAcmDataSet).select("id", "title")
                .withColumnRenamed("id", "idACM");


        Dataset<Row> dblp2WordsData = tokenizer.transform(dblp2DataSet);
        Dataset<Row> hashingDblp2DataSet = hashingTF.transform(dblp2WordsData);
        IDFModel idfDblp2Model = idf.fit(hashingDblp2DataSet);
        Dataset<Row> dblp2Dataset1 = idfDblp2Model.transform(hashingDblp2DataSet).select("id", "title")
                .withColumnRenamed("id", "idDBLP2");

        acmDataset1.crossJoin(dblp2Dataset1).select("idACM", "idDBLP2")
                .show();

    }
}
