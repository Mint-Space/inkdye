package ink.habitats.algorithm;

public class Neuron {

    private int j;
    private int i;
    private double[] input;
    private double[] z;
    private double[] a;
    private double[] bias;
    private double[] forwardPass;
    private double[][] weight;

    private Matrix matrix = new Matrix();
    private ActivationFunction activationFunction = new ActivationFunction();

    public Neuron() {

    }

    public double[] inputFunction(double[] input) {
        return this.input = input;
    }

    public void init(int j, int i) {
        this.i = i;
        this.j = j;
        randomBias(i);
        randomWeight(j, i);
    }

    private double[] randomBias(int biasNumber) {
        return bias = matrix.random(biasNumber);
    }

    private double[][] randomWeight(int row, int column) {
        return weight = matrix.random(row, column);
    }

    private double[] z(double[][] weight, double[] bias) {

        if (weight == null | bias == null) {
            double[][] inputMatrix = matrix.vectorToColumnMatrix(input);
            double[][] product = matrix.multi(matrix.transpose(this.weight), inputMatrix);
            double[] productVecotr = matrix.matrixToVector(product);
            return this.z = matrix.add(productVecotr, this.bias);
        } else {
            this.weight = weight;
            this.bias = bias;
            double[][] inputMatrix = matrix.vectorToColumnMatrix(input);
            double[][] product = matrix.multi(matrix.transpose(weight), inputMatrix);
            double[] productVecotr = matrix.matrixToVector(product);
            return this.z = matrix.add(productVecotr, bias);
        }
    }

    private double[] a(double[] z) {
        return this.a = activationFunction.sigmoid(z);
    }

    public double[] forwardPass(double[][] weight, double[] bias) {
        return forwardPass = a(z(weight, bias));
    }

    public int getJ() {
        return j;
    }

    public void setJ(int j) {
        this.j = j;
    }

    public int getI() {
        return i;
    }

    public void setI(int i) {
        this.i = i;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getZ() {
        return z;
    }

    public void setZ(double[] z) {
        this.z = z;
    }

    public double[] getA() {
        return a;
    }

    public void setA(double[] a) {
        this.a = a;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }

    public double[] getForwardPass() {
        return forwardPass;
    }

    public void setForwardPass(double[] forwardPass) {
        this.forwardPass = forwardPass;
    }

    public double[][] getWeight() {
        return weight;
    }

    public void setWeight(double[][] weight) {
        this.weight = weight;
    }

    public Matrix getMatrix() {
        return matrix;
    }

    public void setMatrix(Matrix matrix) {
        this.matrix = matrix;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
}