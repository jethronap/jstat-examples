package examples.nn.example3;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Example3 {

    public static File baseDir = new File("src/main/resources/datasets/viterbi");
    public static File baseTrainDir = new File(baseDir, "train");
    public static File featuresDirTrain = new File(baseTrainDir, "features");
    public static File labelsDirTrain = new File(baseTrainDir, "labels");
    public static int miniBatchSize = 1;
    public static int  numLabelClasses = 1;

    public static void main(String[] args){

        try{

            Example3 exe = new Example3();
            DataSetIterator data = exe.load_data();

            System.out.println(data.inputColumns());
            DataSet set = data.next(0);

            System.out.println(set.getFeatures());
            System.out.println(set.getLabels());
        }
        catch (Exception e){
            System.out.println("An Exception occurred: ");
            System.out.println("Exception message: " + e.getMessage());
        }
    }

    private DataSetIterator load_data() throws Exception{

        trainFeatures = new CSVSequenceRecordReader(0, ",");
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/features_%d.csv",
                0,0 ));

        //System.out.println(trainFeatures.toString());

        trainLabels = new CSVSequenceRecordReader(0, ",");
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/labels_%d.csv",
                0, 0));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        return trainData;


    }



    SequenceRecordReader trainFeatures = null;
    SequenceRecordReader trainLabels = null;

}
