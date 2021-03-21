package examples.ml_examples.example2;


import jstat.base.Configuration;
import jstat.dataloader.CSVDataLoader;
import jstat.maths.errorfunctions.MSEFunction;
import jstat.ml.trainers.SupervisedTrainer;
import jstat.utils.Pair;
import jstat.optimization.GradientDescent;
import jstat.optimization.GDInput;
import org.nd4j.linalg.api.ndarray.INDArray;
import jstat.ml.regression.NonLinearRegressor;
import jstat.maths.functions.NonLinearVectorPolynomial;
import jstat.maths.functions.ScalarMonomial;


import java.io.IOException;


public class Example2 {

        public static void main(String[] args)throws IOException {

            // set the data directory
            Configuration.dataDirectory = "/home/alex/qi3/jstat/src/main/resources/jstat/datasets/";

            // load data set
            Pair<INDArray, INDArray> dataSet = CSVDataLoader.loadCarPlantWithIntercept(1);

            // assume a hypothesis of the form w0 +w1*X + w2*X^2
            // initially all weights are set o zero
            NonLinearVectorPolynomial hypothesis = new NonLinearVectorPolynomial(new ScalarMonomial(0, 0.0),
                                                                                 new ScalarMonomial(1, 0.0),
                                                                                 new ScalarMonomial(2, 0.0));

            // the regressor
            NonLinearRegressor regression = new NonLinearRegressor(hypothesis);

            // since we do linear regression we will use
            // mean square error as the loss function
            MSEFunction mse = new MSEFunction(regression);

            // configuration parameters for optimizer
            GDInput gdInput = new GDInput();
            gdInput.showIterations = true;
            gdInput.eta = 0.001;
            gdInput.lossFunction = mse;
            gdInput.parameters = regression.getParameters();

            // the optimizer
            GradientDescent gdSolver = new GradientDescent(gdInput);

            // The object responsible for the training
            SupervisedTrainer trainer = new SupervisedTrainer(regression, gdSolver, mse, 50, 1.0e-5);
            trainer.train(dataSet.first, dataSet.second);

            System.out.println("GD Optimization Coefficients "+ regression.getCoeffs());

        }
}
