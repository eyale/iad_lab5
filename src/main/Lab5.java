package src.main;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.functions.SMOreg;
import weka.core.Instance;
import weka.core.Instances;

public class Lab5 {
    public static void main(String args[]) throws Exception {
        Instances data = new Instances(new BufferedReader(new FileReader("src/main/house.arff")));
        data.setClassIndex(data.numAttributes() - 1);
        SMOreg model = new SMOreg();
        model.buildClassifier(data);
        System.out.println(model);
        Instance myHouse = data.lastInstance();
        double price = model.classifyInstance(myHouse);
        System.out.println("My house ("+myHouse+"): "+price);
    }
}
