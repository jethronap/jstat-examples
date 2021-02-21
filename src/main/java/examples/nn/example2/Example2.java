package examples.nn.example2;

import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.convolution.Convolution;


public class Example2 {

    public static void main(String[] args) throws Exception {

        INDArray image = Nd4j.zeros(5, 5);

        for(int i=0; i<image.size(1); ++i) {
            image.putScalar(2, i, 1.);
        }

        System.out.println("Image: " + image);

        INDArray kernel = Nd4j.zeros(3, 3);

        for(int i=0; i<kernel.size(1); ++i) {
            kernel.putScalar(i, 1, 1.);
        }

        //Convolution conv = new Layer.Builder<>(); //ConvolutionLayer.Builder(3).build();
        //INDArray convImage = conv.convn(image, kernel, Convolution.Type.SAME);
        //System.out.println("Convoluted Image: " + convImage);
    }
}
