package ink.habitats.algorithm;

import java.util.LinkedList;

public class Backpropagation {

    private double LearningRate;
    private double[] z;
    private double[] a;
    private double[] b;
    private double[] newBias;
    private double[] output;
    private double[] aOutput;
    private double[] E;
    private double[][] w;
    private double[][] newWeight;
    private LinkedList<double[]> zList;
    private LinkedList<double[]> aList;
    private LinkedList<double[]> bList;
    private LinkedList<double[][]> wList;
    private LinkedList<double[]> newBList = new LinkedList<double[]>();
    private LinkedList<double[][]> newWList = new LinkedList<double[][]>();

    private NeuralNetwork nn;
    private Matrix matrix = new Matrix();
    private ActivationFunction activation = new ActivationFunction();

    public Backpropagation(NeuralNetwork neuralNetwork, double LearningRate) {
        nn = neuralNetwork;
        this.LearningRate = LearningRate;
    }

    public void outputFunction(double[] output) {
        this.output = output;
        getParameter();
        setParameter();
        getAOutput();
    }

    private void getParameter() {
        LinkedList<double[]> zl = nn.getzList();
        zList = (LinkedList<double[]>) zl.clone();
        LinkedList<double[]> al = nn.getaList();
        aList = (LinkedList<double[]>) al.clone();
        LinkedList<double[]> bl = nn.getBiasList();
        bList = (LinkedList<double[]>) bl.clone();
        LinkedList<double[][]> wl = nn.getWeightList();
        wList = (LinkedList<double[][]>) wl.clone();
    }

    private void setParameter() {
        newBList = (LinkedList<double[]>) bList.clone();
        newWList = (LinkedList<double[][]>) wList.clone();
    }

    public double[] getAOutput() {
        return aOutput = aList.get(aList.size() - 1);
    }

    public double[] costFunction() {
        E = matrix.abs(matrix.sub(getAOutput(), output));
        return matrix.multi(matrix.multi(E, 2), 0.5);
    }

    public double costTotal(double[] A) {
        int ar = A.length;
        double result = 0;
        for (int i = 0; i < ar; i++) {
            result += A[i];
        }
        return result /= ar;
    }

    public double[] crossEntropyCost() {
        double[] start = matrix.elementProduct(output, matrix.ln(getAOutput()));
        double[] sub = matrix.sub(1, getAOutput());
        double[] lnSub = matrix.ln(matrix.sub(1, output));
        double[] end = matrix.elementProduct(sub, lnSub);
        double[] result = matrix.add(start, end);
        return result;
    }

    public double[] E() {
        double[] e = matrix.sub(getAOutput(), output);
        double[] er = matrix.elementProduct(matrix.sub(1, getAOutput()), getAOutput());
        return E = matrix.division(e,er);
    }

    public double[][] newWeight(int index) {
        if (index == bList.size() - 1) {
            z = activation.derivationSigmoid(zList.get(index));
            newBias = matrix.elementProduct(z, costFunction());
            newBList.set(index, newBias);
            a = aList.get(index-1);
            double[][] am = matrix.vectorToColumnMatrix(a);
            double[][] newbm = matrix.vectorToRowMatrix(newBias);
            newWeight = matrix.multi(am, newbm);
            newWList.set(index, newWeight);
            return newWeight;
        } else if (index < bList.size()) {
            z = zList.get(index);
            w = wList.get(index + 1);
            if (index > 0) {
                a = aList.get(index - 1);
            } else if (index == 0) {
                a = nn.getX();
            }
            newBias = newBList.get(index + 1);
            double[][] newBiasMatrix = matrix.vectorToColumnMatrix(newBias);
            double[][] newBiasmulti = matrix.multi(w, newBiasMatrix);
            newBias = matrix.matrixToVector(newBiasmulti);
            z = activation.derivationSigmoid(z);
            newBias = matrix.elementProduct(newBias, z);
            newBList.set(index, newBias);
            double[][] am = matrix.vectorToColumnMatrix(a);
            double[][] newbm = matrix.vectorToRowMatrix(newBias);
            newWeight = matrix.multi(am, newbm);
            newWList.set(index, newWeight);
            return newWeight;
        }
        return null;
    }

    public void backwardPass() {
        int l = bList.size() - 1;
        getParameter();
        setParameter();
        for (int i = l; i >= 0; i--) {
            newWeight = newWeight(i);
            w = wList.get(i);
            newWeight = matrix.sub(w, matrix.multi(newWeight, LearningRate));
            newWList.set(i, newWeight);
            b = bList.get(i);
            newBias = matrix.sub(b, matrix.multi(newBias, LearningRate));
            newBList.set(i, newBias);
        }
        update();
        to();
    }

    public void update() {
        nn.setNewBiasList(newBList);
        nn.setNewWeightList(newWList);
    }

    public void to() {
        a = aList.get(aList.size() - 1);
//        System.out.print("Output: ");
//        for (int i = 0; i < a.length; i++) {
//            System.out.print(" " + a[i]);
//        }
//        System.out.println();
    }
}
