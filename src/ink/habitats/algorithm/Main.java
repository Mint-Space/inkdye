package ink.habitats.algorithm;

public class Main {
    private static int max;
    private static int acccuracys;
    private static int errors;
    private static double correctRate;
    private static int total;
    public static void main(String[] args) {
        double[][] X = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        double[] Y = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);
        int[] parameter = { 784, 800, 780, 10 };
        double[] output = new double[10];
        Matrix matrix = new Matrix();
        NeuralNetwork nn = new NeuralNetwork(parameter);
        Backpropagation bp = new Backpropagation(nn, 0.2);
        int xr = X.length;
        for (int i = 0; i < xr; i++) {
            nn.inputFunction(matrix.binarization(X[i], 256));
            nn.forwardPass();
            output = setOutput((int) Y[i]);
            bp.outputFunction(output);
            double[] c = bp.costFunction();
            double ct = bp.costTotal(c);
            max = getMaxIndex(c);
//            System.out.println("第 " + i + " 个 正确答案:  " + Y[i] + " 预测值:" + max + "  CostTotal: " + ct);
            accuracy((int)Y[i],max);
            total = acccuracys+errors;
            bp.backwardPass();
//            System.out.print("Error: ");
//            for (int j = 0; j < c.length; j++) {
//                System.out.print(" " + c[j]);
//            }
//            System.out.println();
        correctRate = acccuracys / Y.length;
        System.out.println("对的： "+acccuracys+" | 错的："+errors +" | 总数："+total);
        }
    }

    public static int accuracy(int correct,int predicted){
        if(correct == predicted){
            return acccuracys ++;
        }else {
            return errors++;
        }
    }

    public static double[] setOutput(int number) {
        double[] result = new double[10];
        result[number] = 1.0;
        return result;
    }

    public static int getMaxIndex(double[] A) {
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
