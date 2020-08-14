package examples.ml_examples.example12;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import jstat.ml.models.HMMConfig;
import jstat.ml.models.HiddenMarkovModel;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Example12 {

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

        // map the sequnce observation to a column index
        Map<String, Integer> obsToIdx = new HashMap<>();
        obsToIdx.put("normal", 0);
        obsToIdx.put("cold",  1);
        obsToIdx.put("dizzy", 2);

        List<String> stateNames = new ArrayList<>();
        stateNames.add("Healthy");
        stateNames.add("Fever");

        HMMConfig config = new HMMConfig();
        config.A = A;
        config.B = B;
        config.pi = pi;
        config.obsToIdx = obsToIdx;
        config.states = stateNames;

        HiddenMarkovModel hmm = new HiddenMarkovModel(config);

        // create a sequence of observations
        List<String> sequence = new ArrayList<String>();
        sequence.add("normal");
        sequence.add("cold");
        sequence.add("dizzy");


        INDArray states = hmm.viterbi(sequence);

        for(int i=0; i<states.size(0); ++i){
            System.out.println(config.states.get(states.getInt(i)));
        }

    }
}
