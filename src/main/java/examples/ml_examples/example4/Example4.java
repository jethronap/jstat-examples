package examples.ml_examples.example4;



import jstat.base.Configuration;
import jstat.dataloader.CSVDataLoader;
import jstat.maths.functions.distances.EuclideanMetric;
import jstat.ml.classifiers.ThreadedKNNClassifier;
import jstat.ml.classifiers.utils.ClassificationVoter;
import jstat.utils.Pair;
import jstat.utils.PairBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import static java.util.concurrent.Executors.newFixedThreadPool;


/** Category: Machine Learning
 * ID: Example8
 * Description: Classification with vanilla ParallelKNN algorithm
 * Taken From:
 * Details:
 * TODO
 */
public class Example4 {

    public static void main(String[] args) throws IOException, IllegalArgumentException{

        // set the data directory
        Configuration.dataDirectory = "/home/alex/qi3/jstat/src/main/resources/jstat/datasets/";

        // load data set
        Pair<INDArray, INDArray> dataSet = CSVDataLoader.loadIrisData();

        ExecutorService executorService = newFixedThreadPool(4);

        System.out.println("Number of rows: "+dataSet.first.size(0));
        System.out.println("Number of labels: "+dataSet.second.size(0));

        List<Pair<Integer, Integer>> partitions = new ArrayList<>(4);
        partitions.add(PairBuilder.makePair(0, 37));
        partitions.add(PairBuilder.makePair(37, 2*37));
        partitions.add(PairBuilder.makePair(2*37, 3*37));
        partitions.add(PairBuilder.makePair(3*37, (int) dataSet.first.size(0)));

        ThreadedKNNClassifier classifier = new ThreadedKNNClassifier(3, false, partitions, executorService);

        classifier.setDistanceCalculator(new EuclideanMetric());
        classifier.setMajorityVoter(new ClassificationVoter());

        classifier.train(dataSet.first, dataSet.second);
        INDArray point = Nd4j.create(new double[]{5.9,3.0,5.1,1.8});

        Integer classIdx = classifier.predictPoint(point);
        System.out.println("Point "+ point +" has class index "+ classIdx);
        executorService.shutdown();

    }
}
