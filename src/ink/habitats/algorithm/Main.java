package ink.habitats.algorithm;

public class Main {
    private static int acccuracys;
    private static int errors;
    private static int[] parameter = { 784, 800, 760, 10 };
    public static void main(String[] args) {
        double[][] X = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        double[] Y = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);
        RunANN runANN = new RunANN(X,Y);
        runANN.setParameter(parameter,0.095);
        int xr = X.length;
        for (int i = 0;i < xr;i++){
            runANN.Run(i);
            double out =  runANN.getOutput();
            if(out == Y[i]){
                acccuracys++;
            }else {
                errors++;
            }
            System.out.println("第 "+(i+1)+"个， OUT："+out+" Y: "+Y[i]+"  对的： "+acccuracys+" | 错的："+errors +" | 正确率："+acccuracys/((double)i+1));
        }
    }
}
