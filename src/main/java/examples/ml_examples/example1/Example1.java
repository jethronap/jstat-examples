package examples.ml_examples.example1;

import jstat.dataloader.CSVDataLoader;
import jstat.base.Configuration;
import jstat.ml.regression.LinearRegressor;
import jstat.ml.trainers.SupervisedTrainer;
import jstat.optimization.GradientDescent;
import jstat.optimization.GDInput;
import jstat.maths.errorfunctions.MSEFunction;

import jstat.utils.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import tech.tablesaw.api.Table;


import java.io.File;
import java.io.IOException;

public class Example1 {


    public static  void main(String[] args ) throws IOException {

        File dataSetFile = new File("/home/alex/qi3/jstat/src/main/resources/jstat/datasets/car_plant.csv");
        Configuration.dataDirectory = dataSetFile;

        Pair<INDArray, INDArray> dataSet = CSVDataLoader.loadCarPlant();

        // the object that represents the
        // linear regression model
        LinearRegressor regression = new LinearRegressor(1);

        // since we do linear regression we will use
        // mean square error as the loss function
        MSEFunction mse = new MSEFunction(regression);

        GDInput gdInput = new GDInput();

        // we will use gradient descent here
        GradientDescent gd = new GradientDescent(gdInput);

        SupervisedTrainer trainer = new SupervisedTrainer(regression, gd, mse, 10, 1.0e-5);
        trainer.train(dataSet.first, dataSet.second);


        //double[] coeffs = regression.getCoeffs();
        //double intercept = regression.getIntercept();
        //System.out.println("Regression coefficients. Intercept: "+intercept+" Slope: "+coeffs[0]);



    }
}
