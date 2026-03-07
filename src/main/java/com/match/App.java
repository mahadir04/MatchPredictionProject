package com.match;

import com.match.preprocessing.DataManager;
import com.match.learning.ModelManager;
import com.match.analysis.AnalysisManager;
import weka.core.Instances;

/**
 * Main entry point for the Match Prediction Pipeline.
 * Using 15k instances and a strict 10-feature set.
 */
public class App {

    private static final String DATASET_PATH = "data/raw/dating_app_behavior_dataset_extended1.csv";
    private static final String MODEL_PATH = "models/match_prediction.model";

    public static void main(String[] args) {
        try {
            System.out.println("==============================================");
            System.out.println("   MATCH PREDICTION & BEHAVIOR ANALYSIS      ");
            System.out.println("==============================================");

            // 1. DATA LOADING
            System.out.println("\n=== DATA LOADING ===");
            Instances rawData = DataManager.loadRawData(DATASET_PATH);
            
            // DISCARD HEAVY FEATURES FIRST
            rawData = DataManager.removeInterestAttribute(rawData);
            
            // Set class index early for stats
            if (rawData.attribute("match_outcome") != null) {
                rawData.setClass(rawData.attribute("match_outcome"));
            } else {
                rawData.setClassIndex(rawData.numAttributes() - 1);
            }

            // 2. DATA DOWNSAMPLING (Optimized for 15k)
            Instances downsampledData = DataManager.downsampleDataset(rawData, 15000);

            // 3. EXPLORATORY ANALYSIS
            DataManager.printExploratoryStats(downsampledData);

            // 4. PREPROCESSING (Target Re-engineering & Strict Feature Selection)
            System.out.println("\n=== PREPROCESSING ===");
            Instances processedData = DataManager.preprocess(downsampledData);
            System.out.println("Preprocessing complete. Target re-engineered to 'is_success'.");
            System.out.println("Strict feature set (10 behavioral attributes) applied.");

            // 5. MODEL TRAINING & EVALUATION
            System.out.println("\n=== MODEL TRAINING & EVALUATION ===");
            ModelManager.evaluateModels(processedData);

            // 6. CLUSTER ANALYSIS
            AnalysisManager.runClustering(processedData);

            // 7. ASSOCIATION RULE MINING
            AnalysisManager.runAssociationRules(processedData);

            // 8. MODEL SERIALIZATION
            System.out.println("\n=== MODEL SERIALIZATION ===");
            ModelManager.saveAndLoadRoundTrip(processedData, MODEL_PATH);

            System.out.println("\n==============================================");
            System.out.println("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY ");
            System.out.println("==============================================");

        } catch (Exception e) {
            System.err.println("\n[ERROR] Pipeline failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
