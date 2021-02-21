package examples.miscellaneous.example1;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


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
        //INDArray B = Nd4j.zeros(2, 3);
        INDArray B = Nd4j.create(new double[][]{{0.6, 0.4, 0.5},{0.6, 0.4, 0.3}});

        System.out.println("Shape-size: " + B.shape().length);
        System.out.println("Rows: " + B.size(0));
        System.out.println("Columns: " + B.size(1));
        System.out.println("Column 1: " + B.getColumn(2));
        argMax = Nd4j.argMin(B.getColumn(1), 0).getInt(0);
        System.out.println("ArgMax : " + argMax);

    }
}
