package examples.ml_examples.example11;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import jstat.ml.models.HMMHelpers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Example11 {

    public static void main(String[] args){

        // create the transition probability matrix
        int nRows = 2;
        int nColumns = 2;
        INDArray A = Nd4j.zeros(nRows, nColumns);

        A.putScalar(0,0, 0.7);
        A.putScalar(0,1, 0.3);
        A.putScalar(1,0, 0.4);
        A.putScalar(1,1, 0.6);

        // create the emission probability matrix
        INDArray B = Nd4j.zeros(2, 3);

        B.putScalar(0,0, 0.5);
        B.putScalar(0,1, 0.4);
        B.putScalar(0,2, 0.1);
        B.putScalar(1,0, 0.1);
        B.putScalar(1,1, 0.3);
        B.putScalar(1,2, 0.6);


        // create the initialization vector
        INDArray pi = Nd4j.create(new double[]{0.6, 0.4});

        // create a sequence of observations
        List<String> sequence = new ArrayList<String>();
        sequence.add("a");
        sequence.add("b");
        sequence.add("c");

        // map the sequnce observation to a column index
        Map<String, Integer> obsToIdx = new HashMap<>();
        obsToIdx.put("a", 0);
        obsToIdx.put("b", 1);
        obsToIdx.put("c", 2);

        // compute alpha
        INDArray beta = HMMHelpers.backward(sequence, A, B, pi, obsToIdx);

        System.out.println("beta matrix: ");
        System.out.println(beta);

        // we can now calculate the probability
        double p = 0.0;
        for(int i=0; i<A.shape()[0]; ++i){
            p += pi.getDouble(0, i)*B.getDouble(i, obsToIdx.get(sequence.get(0)))*beta.getDouble(1, i);
        }

        System.out.println("probability: " + p);
    }
}
