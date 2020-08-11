package examples.ml_examples.example10;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import ml.models.HMMHelpers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Example10 {

    public static void main(String[] args){

        // create the transition probability matrix
        int nRows = 3;
        int nColumns = 3;
        INDArray A = Nd4j.zeros(nRows, nColumns);

        A.putScalar(0,0, 0.5);
        A.putScalar(0,1, 0.25);
        A.putScalar(0,2, 0.25);
        A.putScalar(1,0, 0.1);
        A.putScalar(1,1, 0.8);
        A.putScalar(1,2, 0.1);
        A.putScalar(2,0, 0.3);
        A.putScalar(2,1, 0.15);
        A.putScalar(2,2, 0.6);

        // create the emission probability matrix
        INDArray B = Nd4j.zeros(3, 3);

        B.putScalar(0,0, 0.5);
        B.putScalar(0,1, 0.25);
        B.putScalar(0,2, 0.25);
        B.putScalar(1,0, 0.1);
        B.putScalar(1,1, 0.8);
        B.putScalar(1,2, 0.1);
        B.putScalar(2,0, 0.3);
        B.putScalar(2,1, 0.15);
        B.putScalar(2,2, 0.6);

        // create the initialization vector
        INDArray pi = Nd4j.create(new double[]{0.7, 0.15, 0.15});

        List<String> sequence = new ArrayList<String>();
        sequence.add("a");
        sequence.add("b");
        sequence.add("a");
        sequence.add("c");
        sequence.add("b");
        sequence.add("a");
        Map<String, Integer> obsToIdx = new HashMap<>();
        obsToIdx.put("a", 0);
        obsToIdx.put("b", 1);
        obsToIdx.put("c", 2);

        INDArray alpha = HMMHelpers.forward(sequence, A, B, pi, obsToIdx);

        System.out.println(alpha);

    }
}
