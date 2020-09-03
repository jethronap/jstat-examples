package examples.ml_examples.example9;

import org.nd4j.linalg.api.ndarray.INDArray;
import jstat.ml.models.HiddenMarkovModel;
import jstat.ml.models.HMMConfig;
import org.nd4j.linalg.factory.Nd4j;

public class Example9 {

    public static void main(String[] args){

        // create the transition probability matrix
        int nRows = 2;
        int nColumns = 2;
        INDArray A = Nd4j.zeros(nRows, nColumns);

        // create the emission probability matrix
        INDArray B = Nd4j.zeros(2, 3);

        // create the initialization vector
        INDArray pi = Nd4j.zeros(nRows);

        HMMConfig config = new HMMConfig();
        // the Hidden Markov Model
        HiddenMarkovModel hmm = new HiddenMarkovModel(config);

        //hmm.viterbi()



    }
}
