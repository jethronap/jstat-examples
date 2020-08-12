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

        B.putScalar(0,0, 0.16);
        B.putScalar(0,1, 0.26);
        B.putScalar(0,2, 0.58);
        B.putScalar(1,0, 0.25);
        B.putScalar(1,1, 0.28);
        B.putScalar(1,2, 0.47);
        B.putScalar(2,0, 0.2);
        B.putScalar(2,1, 0.1);
        B.putScalar(2,2, 0.7);

        // create the initialization vector
        INDArray pi = Nd4j.create(new double[]{0.7, 0.15, 0.15});

        // create a sequence of observations
        List<String> sequence = new ArrayList<String>();
        sequence.add("a");
        sequence.add("b");
        sequence.add("a");
        sequence.add("c");
        sequence.add("b");
        sequence.add("a");

        // map the sequnce observation to a column index
        Map<String, Integer> obsToIdx = new HashMap<>();
        obsToIdx.put("a", 0);
        obsToIdx.put("b", 1);
        obsToIdx.put("c", 2);

        // compute alpha
        INDArray alpha = HMMHelpers.forward(sequence, A, B, pi, obsToIdx);

        System.out.println("alpha matrix: ");
        System.out.println(alpha);

        // we can now calculate the probability
        double p = 0.0;
        for(int i=0; i<A.shape()[0]; ++i){
            p += alpha.getDouble(sequence.size()-1, i);
        }

        System.out.println("probability: " + p);
    }
}
