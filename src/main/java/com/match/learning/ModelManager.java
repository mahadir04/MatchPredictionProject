package com.match.learning;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.supervised.instance.SMOTE;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.SerializationHelper;
import com.match.preprocessing.DataManager;

import java.io.File;
import java.util.Random;

public class ModelManager {

    public static void evaluateModels(Instances data) throws Exception {
        System.out.println("\n=== CLASS BALANCING ===");
        System.out.println("Applying SMOTE inside cross-validation folds.");

        // Define Base Classifiers
        J48 j48 = new J48();
        
        SMO smo = new SMO();

        NaiveBayes nb = new NaiveBayes();

        Bagging bagging = new Bagging();
        bagging.setClassifier(new J48());
        bagging.setNumIterations(10);

        AdaBoostM1 adaboost = new AdaBoostM1();
        adaboost.setClassifier(new J48());
        adaboost.setNumIterations(10); 

        // 1. Naive Bayes with SMOTE
        SMOTE smote = new SMOTE();
        smote.setPercentage(100);
        
        FilteredClassifier nbFiltered = new FilteredClassifier();
        nbFiltered.setFilter(smote);
        nbFiltered.setClassifier(nb);

        // 2. Bagging (J48) with SMOTE
        FilteredClassifier bagFiltered = new FilteredClassifier();
        bagFiltered.setFilter(smote);
        bagFiltered.setClassifier(bagging);

        // 3. AdaBoost (J48) with SMOTE
        FilteredClassifier adaFiltered = new FilteredClassifier();
        adaFiltered.setFilter(smote);
        adaFiltered.setClassifier(adaboost);

        // 4. SMO with Normalize (Critical for SVM)
        Normalize normalize = new Normalize();
        FilteredClassifier smoFiltered = new FilteredClassifier();
        smoFiltered.setFilter(normalize);
        smoFiltered.setClassifier(smo);

        Classifier[] models = {nbFiltered, bagFiltered, adaFiltered, smoFiltered};
        String[] modelNames = {"NAIVE BAYES", "BAGGING (J48)", "ADABOOST (J48)", "SMO (SVM)"};

        System.out.println("\n=== MODEL TRAINING & EVALUATION ===");

        for (int i = 0; i < models.length; i++) {
            System.out.println("\n=== " + modelNames[i] + " RESULTS ===");
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(models[i], data, 10, new Random(1));

            System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());
            System.out.printf("Precision: %.2f\n", eval.weightedPrecision());
            System.out.printf("Recall: %.2f\n", eval.weightedRecall());
            System.out.printf("F1 Score: %.2f\n", eval.weightedFMeasure());
            System.out.printf("ROC AUC: %.2f\n", eval.weightedAreaUnderROC());

            System.out.println("\nConfusion Matrix:");
            System.out.println(eval.toMatrixString());
            System.out.println("-----------------------------");
        }
    }

    public static void saveAndLoadRoundTrip(Instances data, String modelPath) throws Exception {
        System.out.println("\n=== MODEL SERIALIZATION (Round-Trip) ===");
        
        Bagging bestModel = new Bagging();
        bestModel.setClassifier(new J48());
        bestModel.buildClassifier(data);
        
        new File("models").mkdirs();
        SerializationHelper.write(modelPath, bestModel);
        System.out.println("Model saved successfully: " + modelPath);

        Classifier loadedModel = (Classifier) SerializationHelper.read(modelPath);
        System.out.println("Model loaded successfully: " + modelPath);

        Instance firstInstance = data.instance(0);
        double actual = firstInstance.classValue();
        double prediction = loadedModel.classifyInstance(firstInstance);

        System.out.println("\n--- Model Verification ---");
        System.out.println("Actual Class: " + firstInstance.classAttribute().value((int) actual));
        System.out.println("Predicted Class: " + firstInstance.classAttribute().value((int) prediction));
    }
}
