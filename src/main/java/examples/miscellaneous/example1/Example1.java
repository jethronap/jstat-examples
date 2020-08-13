package examples.miscellaneous.example1;

import ml.models.HMMHelpers;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Example1 {

    public static void main(String[] args){

        // create an 1D array
        INDArray A = Nd4j.zeros(10);

        System.out.println("Shape-size: " + A.shape().length);
        System.out.println("size: " + A.shape()[0]);
        System.out.println("Other size: " + A.size(0));

        for(int i=0; i<A.size(0); ++i) {
            A.putScalar(i, (double) i);
        }

        double max = Nd4j.max(A).getDouble(0);
        long argMax = Nd4j.argMax(A, 0).getInt(0);
        System.out.println("Max : " + max);
        System.out.println("ArgMax : " + argMax);

        double min = Nd4j.min(A).getDouble(0);
        long argMin = Nd4j.argMin(A, 0).getInt(0);
        System.out.println("Min: " + min);
        System.out.println("ArgMin : " + argMin);

        // create a 2D matrix
        /*INDArray B = Nd4j.zeros(2, 3);

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

        System.out.println("probability: " + p);*/
    }
}
