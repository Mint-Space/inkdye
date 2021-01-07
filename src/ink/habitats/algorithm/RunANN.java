package ink.habitats.algorithm;

public class RunANN {

    private static int acccuracys;
    private static int errors;

    private double[][] X;
    private double[] Y;
    private double[] C;
    private int[] neuralNumberList;
    private double LearningRate;
    private int outputResult;
    private NeuralNetwork neuralNetwork;
    private Backpropagation backPropagation;

    public RunANN(double[][] X, double[] Y) {
        this.X = X;
        this.Y = Y;
    }

    public void Run(int index) {
        if (index > 0) {
            neuralNetwork.inputFunction(new Matrix().binarization(X[index], 256));
            neuralNetwork.forwardPass();
            backPropagation.outputFunction(setOutput((int) Y[index]));
            C = backPropagation.costFunction();
            backPropagation.costTotal(C);
            outputResult = getMaxIndex(C);
            backPropagation.backwardPass();
        } else {
            System.out.println("数据集为空！ 索引值为：" + index);
        }
    }

    public double getOutput(){
        return outputResult;
    }
    public int accuracy(int correct, int predicted) {
        if (correct == predicted) {
            return acccuracys++;
        } else {
            return errors++;
        }
    }

    public void setParameter(int[] neuralNumberList, double LearningRate) {
        this.neuralNumberList = neuralNumberList;
        this.LearningRate = LearningRate;
        neuralNetwork = new NeuralNetwork(neuralNumberList);
        backPropagation = new Backpropagation(neuralNetwork, LearningRate);
    }

    public double[] setOutput(int number) {
        double[] result = new double[neuralNumberList[neuralNumberList.length - 1]];
        result[number] = 1.0;
        return result;
    }

    public int getMaxIndex(double[] A) {
        int ar = A.length;
        int maxIndex = -1;
        double max = 0;
        for (int i = 0; i < ar; i++) {
            if (A[i] > max) {
                max = A[i];
                maxIndex = i;
            } else if (A[i] < max) {
                max = max;
                maxIndex = maxIndex;
            }
        }
        return maxIndex;
    }
}
