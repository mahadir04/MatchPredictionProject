package matchpredictionproject;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.clusterers.SimpleKMeans;

import weka.associations.Apriori;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

public class MatchPredictionProject {

    public static void main(String[] args) {

        try {

            // ==============================
            // 1. LOAD DATASET
            // ==============================

            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("dating_app_behavior_dataset_extended1.csv"));
            Instances rawData = loader.getDataSet();

            System.out.println("Dataset Loaded: " + rawData.numInstances() + " instances");



            // ==============================
            // 2. CREATE BINARY MATCH COLUMN
            // ==============================

            ArrayList<String> classValues = new ArrayList<>();
            classValues.add("No");
            classValues.add("Yes");

            Attribute matchAttr = new Attribute("match", classValues);

            rawData.insertAttributeAt(matchAttr, rawData.numAttributes());

            for (int i = 0; i < rawData.numInstances(); i++) {

                String outcome = rawData.instance(i)
                        .stringValue(rawData.attribute("match_outcome"));

                if (outcome.equals("Mutual Match") ||
                    outcome.equals("Date Happened") ||
                    outcome.equals("Relationship Formed") ||
                    outcome.equals("Instant Match")) {

                    rawData.instance(i).setValue(rawData.numAttributes() - 1, "Yes");

                } else {

                    rawData.instance(i).setValue(rawData.numAttributes() - 1, "No");
                }
            }



            /// ==========================
            // 3. KEEP ONLY IMPORTANT ATTRIBUTES
            // ==========================

            Remove remove = new Remove();

            String keepAttributes =
                    "likes_received,"
                  + "mutual_matches,"
                  + "message_sent_count,"
                  + "swipe_right_ratio,"
                  + "bio_length,"
                  + "profile_pics_count,"
                  + "age,"
                  + "relationship_intent,"
                  + "interest_tags,"
                  + "education_level,"
                  + "income_bracket,"
                  + "gender,"
                  + "sexual_orientation,"
                  + "last_active_hour,"
                  + "emoji_usage_rate,"
                  + "body_type,"
                  + "match_outcome";

            String[] attrNames = keepAttributes.split(",");

            String indices = "";

            for (String name : attrNames) {

                Attribute attr = rawData.attribute(name.trim());

                if (attr != null) {
                    indices += (attr.index() + 1) + ",";
                }
            }

            indices = indices.substring(0, indices.length() - 1);

            remove.setAttributeIndices(indices);
            remove.setInvertSelection(true);
            remove.setInputFormat(rawData);

            Instances data = Filter.useFilter(rawData, remove);

            data.setClass(data.attribute("match_outcome"));

            System.out.println("Selected Important Attributes Only");



            // ==============================
            // 4. SET CLASS ATTRIBUTE
            // ==============================

            data.setClassIndex(data.numAttributes() - 1);



            // ==============================
            // 5. HANDLE MISSING VALUES
            // ==============================

            ReplaceMissingValues replace = new ReplaceMissingValues();
            replace.setInputFormat(data);
            data = Filter.useFilter(data, replace);



            // ==============================
            // 6. SUPERVISED LEARNING
            // ==============================

            System.out.println("\n=== SUPERVISED LEARNING RESULTS ===");

            weka.classifiers.Classifier[] models = {
                    new RandomForest(),
                    new J48(),
                    new SMO()
            };

            String[] modelNames = {
                    "RandomForest",
                    "J48 Decision Tree",
                    "SMO (SVM)"
            };

            for (int i = 0; i < models.length; i++) {

                Evaluation eval = new Evaluation(data);

                eval.crossValidateModel(models[i], data, 10, new Random(1));

                System.out.println(modelNames[i] +
                        " Accuracy: " +
                        String.format("%.2f", eval.pctCorrect()) + "%");

                System.out.println("Precision: " + eval.weightedPrecision());
                System.out.println("Recall: " + eval.weightedRecall());
                System.out.println("F1 Score: " + eval.weightedFMeasure());
                System.out.println("-----------------------------");
            }



            // ==============================
            // 7. ADVANCED CLUSTERING
            // ==============================

            System.out.println("\n=== CLUSTERING ANALYSIS ===");

            Remove removeClass = new Remove();
            removeClass.setAttributeIndices("" + (data.classIndex() + 1));
            removeClass.setInputFormat(data);

            Instances clusterData = Filter.useFilter(data, removeClass);
            clusterData.setClassIndex(-1);



            // Normalize data
            Normalize normalize = new Normalize();
            normalize.setInputFormat(clusterData);
            clusterData = Filter.useFilter(clusterData, normalize);



            // Test different K values
            double bestError = Double.MAX_VALUE;
            int bestK = 2;

            System.out.println("\nTesting K values:");

            for (int k = 2; k <= 6; k++) {

                SimpleKMeans kmeans = new SimpleKMeans();

                kmeans.setNumClusters(k);
                kmeans.setSeed(10);
                kmeans.setPreserveInstancesOrder(true);

                kmeans.buildClusterer(clusterData);

                double error = kmeans.getSquaredError();

                System.out.println("K = " + k + " | SSE = " + error);

                if (error < bestError) {

                    bestError = error;
                    bestK = k;
                }
            }

            System.out.println("\nBest K Value: " + bestK);



            // Final KMeans model
            SimpleKMeans finalKMeans = new SimpleKMeans();

            finalKMeans.setNumClusters(bestK);
            finalKMeans.setSeed(10);
            finalKMeans.setPreserveInstancesOrder(true);

            finalKMeans.buildClusterer(clusterData);

            System.out.println("\nCluster Centroids:");
            System.out.println(finalKMeans.getClusterCentroids());



            int[] assignments = finalKMeans.getAssignments();
            int[] clusterSizes = new int[bestK];

            for (int i = 0; i < assignments.length; i++) {
                clusterSizes[assignments[i]]++;
            }

            System.out.println("\nCluster Sizes:");
            for (int i = 0; i < clusterSizes.length; i++) {
                System.out.println("Cluster " + i + ": " + clusterSizes[i] + " instances");
            }

            System.out.println("\nCluster Sizes:");

            for (int i = 0; i < clusterSizes.length; i++) {

                System.out.println("Cluster " + i + ": " + clusterSizes[i]);
            }



          


            // ==============================
            // 9. ASSOCIATION RULE MINING
            // ==============================

            System.out.println("\n=== ASSOCIATION RULES (Apriori) ===");

            Discretize disc = new Discretize();

            disc.setInputFormat(data);

            Instances discData = Filter.useFilter(data, disc);

            discData.setClassIndex(-1);

            Apriori apriori = new Apriori();

            apriori.setNumRules(5);

            apriori.buildAssociations(discData);

            System.out.println(apriori);



            // ==============================
            // 10. SAVE BEST MODEL
            // ==============================

            RandomForest bestModel = new RandomForest();

            bestModel.buildClassifier(data);

            weka.core.SerializationHelper.write("match_prediction.model", bestModel);

            System.out.println("\nModel saved successfully as: match_prediction.model");



        } catch (Exception e) {

            System.out.println("Execution Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}