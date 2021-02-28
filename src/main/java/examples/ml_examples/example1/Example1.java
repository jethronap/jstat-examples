package examples.ml_examples.example1;

import jstat.dataloader.CSVDataLoader;
import jstat.base.Configuration;
import jstat.ml.regression.LinearRegressor;
import jstat.ml.trainers.SupervisedTrainer;
import jstat.optimization.GradientDescent;
import jstat.optimization.GDInput;
import jstat.maths.errorfunctions.MSEFunction;

import jstat.optimization.OLSOptimizer;
import jstat.utils.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class Example1 {


    public static  void main(String[] args ) throws IOException {

        Configuration.dataDirectory = "/home/alex/qi3/jstat/src/main/resources/jstat/datasets/";

        Pair<INDArray, INDArray> dataSet = CSVDataLoader.loadCarPlantWithIntercept();

        // the object that represents the
        // linear regression model
        LinearRegressor regression = new LinearRegressor(1, true);

        // since we do linear regression we will use
        // mean square error as the loss function
        MSEFunction mse = new MSEFunction(regression);

        GDInput gdInput = new GDInput();
        gdInput.lossFunction = mse;
        gdInput.parameters = regression.getParameters();
        gdInput.eta = 0.01;

        // we will use gradient descent here
        GradientDescent gd = new GradientDescent(gdInput);

        SupervisedTrainer trainer = new SupervisedTrainer(regression, gd, mse, 50, 1.0e-5);
        trainer.train(dataSet.first, dataSet.second);

        System.out.println("GD Optimization Coefficients "+ regression.getCoeffs());

        // do an ordinary least squares to check the solution
        INDArray params = Nd4j.zeros(2);
        OLSOptimizer olsOptimizer = new OLSOptimizer(params);
        olsOptimizer.step(dataSet.first, dataSet.second);
        System.out.println("OLS Optimization coefficients "+ params);

    }
}
