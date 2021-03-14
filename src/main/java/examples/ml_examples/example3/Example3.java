package examples.ml_examples.example3;

import jstat.maths.functions.distances.EuclideanMetric;
import jstat.ml.classifiers.KNNClassifier;
import jstat.ml.classifiers.utils.ClassificationVoter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class Example3 {

    public static void main(String[] args) {

        INDArray dataSet = Nd4j.zeros(12, 2);

        dataSet.put(0, 0, 1.0);
        dataSet.put(0, 1, 3.0);
        dataSet.put(1, 0, 1.5);
        dataSet.put(1, 1,2.0);
        dataSet.put(2, 0, 2.0);
        dataSet.put(2,1,1.0);
        dataSet.put(3, 0, 2.5);
        dataSet.put(3, 1, 4.0);
        dataSet.put(4, 0, 3.0);
        dataSet.put(4, 1, 1.5);
        dataSet.put(5, 0, 3.5);
        dataSet.put(5, 1,2.5);
        dataSet.put(6, 0, 5.0);
        dataSet.put(6, 1, 5.0);
        dataSet.put(7, 0, 5.5);
        dataSet.put(7, 1, 4.0);
        dataSet.put(8, 0, 6.0);
        dataSet.put(8, 1,6.0);
        dataSet.put(9, 0, 6.5);
        dataSet.put(9, 1, 4.5);
        dataSet.put(10, 0, 7.0);
        dataSet.put(10, 1,1.5);
        dataSet.put(11, 0, 8.0);
        dataSet.put(11, 1, 2.5);

        List labels = new ArrayList(12);

        for (int i = 0; i < 6; ++i) {
            labels.add(0);
        }

        for (int i = 6; i < labels.size(); ++i) {
            labels.add(1);
        }


        KNNClassifier classifier = new KNNClassifier(2, false);

        classifier.setDistanceCalculator(new EuclideanMetric());
        classifier.setMajorityVoter(new ClassificationVoter());

        classifier.train(dataSet, labels);

        INDArray point = Nd4j.zeros(1, 2);
        point.putScalar(0, 0, 3.1);
        point.putScalar(0, 1, 2.2);

        Integer classIdx = classifier.predictPoint(point);

        System.out.println("Point " + point + " has class index " + classIdx);
        point.putScalar(0, 0, 9.1);
        point.putScalar(0, 1, 6.2);

        classIdx = classifier.predictPoint(point);
        System.out.println("Point " + point + " has class index " + classIdx);
    }
}
