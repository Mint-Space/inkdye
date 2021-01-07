package ink.habitats.algorithm;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedList;

public class NeuralNetwork {

    int k = 0;
    private int L;
    private double[] X;
    private int[] neuronParameter;
    private int[][] weightParameter;
    private double[] z;
    private double[] a;
    private double[] bias;
    private double[] newBias;
    private double[] forwardPass;
    private double[][] weight;
    private double[][] newWeight;

    private LinkedList<Neuron> neuronList = new LinkedList<Neuron>();
    private LinkedList<double[]> zList = new LinkedList<double[]>();
    private LinkedList<double[]> aList = new LinkedList<double[]>();
    private LinkedList<double[]> biasList = new LinkedList<double[]>();
    private LinkedList<double[][]> weightList = new LinkedList<double[][]>();
    private LinkedList<double[]> newBiasList = new LinkedList<double[]>();
    private LinkedList<double[][]> newWeightList = new LinkedList<double[][]>();
    private LinkedList<double[]> forwardPassList = new LinkedList<double[]>();

    private Neuron neuron;

    public NeuralNetwork(int[] neuronParameter) {
        this.neuronParameter = neuronParameter;
        creatNeuralNetwork();
    }

    public void inputFunction(double[] X) {
        neuron = neuronList.get(0);
        this.X = neuron.inputFunction(X);
    }

    private void creatNeuralNetwork() {
        L = neuronParameter.length;
        for (int i = 0; i < L - 1; i++) {
            neuron = new Neuron();
            neuron.init(neuronParameter[i], neuronParameter[i + 1]);
            bias = neuron.getBias();
            weight = neuron.getWeight();
            biasList.add(bias);
            weightList.add(weight);
            neuronList.add(neuron);
        }
    }

    public void forwardPass() {
        for (int i = 0; i < L - 1; i++) {
            neuron = neuronList.get(i);
            if (i >= 1) {
                neuron = neuronList.get(i);
                neuron.setInput(forwardPass);
            }
            if (k > 0) {
                weight = neuron.getWeight();
                bias = neuron.getBias();
                weightList.set(i, weight);
                biasList.set(i, bias);
                update(i);
            }
            forwardPass = neuron.forwardPass(newWeight, newBias);
            if (k == 0) {
                z = neuron.getZ();
                a = neuron.getA();
                zList.add(z);
                aList.add(a);
                forwardPassList.add(forwardPass);
            } else {
                z = neuron.getZ();
                a = neuron.getA();
                zList.set(i, z);
                aList.set(i, a);
                forwardPassList.set(i, forwardPass);
            }
        }
        k++;
    }

    public void update(int index) {
        newBias = newBiasList.get(index);
        newWeight = newWeightList.get(index);
    }

    public int getL() {
        return L;
    }

    public void setL(int l) {
        L = l;
    }

    public double[] getX() {
        return X;
    }

    public void setX(double[] x) {
        X = x;
    }

    public int[] getNeuronParameter() {
        return neuronParameter;
    }

    public void setNeuronParameter(int[] neuronParameter) {
        this.neuronParameter = neuronParameter;
    }

    public int[][] getWeightParameter() {
        return weightParameter;
    }

    public void setWeightParameter(int[][] weightParameter) {
        this.weightParameter = weightParameter;
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

    public double[] getNewBias() {

        return newBias;
    }

    public void setNewBias(double[] newBias) {

        this.newBias = newBias;
    }

    public double[][] getWeight() {
        return weight;
    }

    public void setWeight(double[][] weight) {
        this.weight = weight;
    }

    public double[][] getNewWeight() {
        return newWeight;
    }

    public void setNewWeight(double[][] newWeight) {
        this.newWeight = newWeight;
    }

    public LinkedList<Neuron> getNeuronList() {
        return neuronList;
    }

    public void setNeuronList(LinkedList<Neuron> neuronList) {
        this.neuronList = neuronList;
    }

    public LinkedList<double[]> getzList() {
        return zList;
    }

    public void setzList(LinkedList<double[]> zList) {
        this.zList = zList;
    }

    public LinkedList<double[]> getaList() {
        return aList;
    }

    public void setaList(LinkedList<double[]> aList) {
        this.aList = aList;
    }

    public LinkedList<double[]> getBiasList() {
        return biasList;
    }

    public void setBiasList(LinkedList<double[]> biasList) {
        this.biasList = biasList;
    }

    public LinkedList<double[][]> getWeightList() {
        return weightList;
    }

    public void setWeightList(LinkedList<double[][]> weightList) {
        this.weightList = weightList;
    }

    public Neuron getNeuron() {
        return neuron;
    }

    public void setNeuron(Neuron neuron) {
        this.neuron = neuron;
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public LinkedList<double[]> getNewBiasList() {
        return newBiasList;
    }

    public void setNewBiasList(LinkedList<double[]> newBiasList) {
        this.newBiasList = newBiasList;
    }

    public LinkedList<double[][]> getNewWeightList() {
        return newWeightList;
    }

    public void setNewWeightList(LinkedList<double[][]> newWeightList) {
        this.newWeightList = newWeightList;
    }

    public double[] getForwardPass() {
        return forwardPass;
    }

    public void setForwardPass(double[] forwardPass) {
        this.forwardPass = forwardPass;
    }

    public LinkedList<double[]> getForwardPassList() {
        return forwardPassList;
    }

    public void setForwardPassList(LinkedList<double[]> forwardPassList) {
        this.forwardPassList = forwardPassList;
    }

}

