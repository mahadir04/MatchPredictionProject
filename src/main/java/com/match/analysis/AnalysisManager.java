package com.match.analysis;

import weka.associations.Apriori;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

public class AnalysisManager {

    public static void runClustering(Instances data) throws Exception {
        System.out.println("\n=== CLUSTER ANALYSIS ===");
        
        Remove removeClass = new Remove();
        removeClass.setAttributeIndices("" + (data.classIndex() + 1));
        removeClass.setInputFormat(data);
        Instances clusterData = Filter.useFilter(data, removeClass);

        Normalize normalize = new Normalize();
        normalize.setInputFormat(clusterData);
        clusterData = Filter.useFilter(clusterData, normalize);

        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.setSeed(10);
        kmeans.setPreserveInstancesOrder(true);
        kmeans.buildClusterer(clusterData);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(kmeans);
        eval.evaluateClusterer(clusterData);

        System.out.println("Clusters discovered: " + kmeans.getNumClusters());
        
        double[] assignments = eval.getClusterAssignments();
        int[] clusterSizes = new int[kmeans.getNumClusters()];
        for (double a : assignments) {
            clusterSizes[(int) a]++;
        }

        for (int i = 0; i < clusterSizes.length; i++) {
            System.out.println("Cluster " + i + " \u2192 " + clusterSizes[i] + " users");
        }

        System.out.println("\nCentroids (Representative Profiles):");
        System.out.println(kmeans.getClusterCentroids());
    }

    public static void runAssociationRules(Instances data) throws Exception {
        System.out.println("\n=== ASSOCIATION RULE MINING ===");

        // Use broad discretization (3 bins: Low, Medium, High) to find more rules
        Discretize disc = new Discretize();
        disc.setBins(3); 
        disc.setInputFormat(data);
        Instances discData = Filter.useFilter(data, disc);

        discData.setClassIndex(discData.numAttributes() - 1);

        Apriori apriori = new Apriori();
        apriori.setNumRules(20);
        apriori.setMinMetric(0.5); // Lowered Confidence to 50%
        apriori.setLowerBoundMinSupport(0.02); // Lowered Support to 2%
        apriori.setCar(true); // Class Association Rules

        apriori.buildAssociations(discData);

        System.out.println("Top Rules Discovered:");
        String rules = apriori.toString();
        if (rules.contains("No rules found")) {
            System.out.println("Status: No rules found with current thresholds. Try further lowering support/confidence.");
        } else {
            System.out.println(rules);
        }
    }
}
