package com.match.preprocessing;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.supervised.instance.SMOTE;

import java.io.File;
import java.util.Random;

public class DataManager {

    public static Instances loadRawData(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();
        System.out.println("Dataset Loaded: " + data.numInstances() + " instances.");
        return data;
    }

    public static Instances downsampleDataset(Instances data, int maxRows) {
        System.out.println("\n=== DATASET SIZE CHECK ===");
        System.out.println("Original instances: " + data.numInstances());
        
        if (data.numInstances() <= maxRows) {
            System.out.println("Dataset size is within limits. No downsampling needed.");
            return data;
        }

        System.out.println("Downsampling to: " + maxRows);
        data.randomize(new Random(42));
        Instances downsampled = new Instances(data, 0, maxRows);
        System.out.println("Final dataset size: " + downsampled.numInstances());
        return downsampled;
    }

    public static Instances removeInterestAttribute(Instances data) throws Exception {
        System.out.println("Manually discarding 'interest_tags' feature to optimize memory...");
        Attribute interestAttr = data.attribute("interest_tags");
        if (interestAttr != null) {
            Remove remove = new Remove();
            remove.setAttributeIndices("" + (interestAttr.index() + 1));
            remove.setInputFormat(data);
            return Filter.useFilter(data, remove);
        }
        return data;
    }

    public static void printExploratoryStats(Instances data) {
        System.out.println("\n=== EXPLORATORY DATA ANALYSIS ===");
        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());

        // Class Distribution
        if (data.classIndex() != -1) {
            System.out.println("\nClass Distribution (" + data.classAttribute().name() + "):");
            AttributeStats stats = data.attributeStats(data.classIndex());
            for (int i = 0; i < data.classAttribute().numValues(); i++) {
                System.out.printf("  %-10s: %d\n", data.classAttribute().value(i), stats.nominalCounts[i]);
            }
        }

        System.out.println("\nNumeric Attributes:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr.isNumeric()) {
                AttributeStats stats = data.attributeStats(i);
                System.out.printf("  %-20s \u2192 mean: %-8.2f | min: %-8.2f | max: %-8.2f\n",
                        attr.name(), stats.numericStats.mean, stats.numericStats.min, stats.numericStats.max);
            }
        }

        System.out.println("\nNominal Attributes:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr.isNominal() && i != data.classIndex()) {
                System.out.println("  " + attr.name());
                AttributeStats stats = data.attributeStats(i);
                for (int j = 0; j < attr.numValues(); j++) {
                    if (stats.nominalCounts[j] > 0) {
                        System.out.printf("    %-15s: %d\n", attr.value(j), stats.nominalCounts[j]);
                    }
                }
            }
        }
    }

    public static Instances preprocess(Instances rawData) throws Exception {
        // --- TARGET RE-ENGINEERING FOR ACCURACY ---
        java.util.ArrayList<String> classValues = new java.util.ArrayList<>();
        classValues.add("No");
        classValues.add("Yes");
        Attribute successAttr = new Attribute("is_success", classValues);
        rawData.insertAttributeAt(successAttr, rawData.numAttributes());

        for (int i = 0; i < rawData.numInstances(); i++) {
            String outcome = rawData.instance(i).stringValue(rawData.attribute("match_outcome"));
            if (outcome.equals("Mutual Match") || outcome.equals("Date Happened") || 
                outcome.equals("Relationship Formed") || outcome.equals("Instant Match")) {
                rawData.instance(i).setValue(rawData.numAttributes() - 1, "Yes");
            } else {
                rawData.instance(i).setValue(rawData.numAttributes() - 1, "No");
            }
        }

        // --- STRICT FEATURE SELECTION (Only the requested 10 + target) ---
        Remove remove = new Remove();
        String keepAttributes = "likes_received,mutual_matches,message_sent_count,swipe_right_ratio,profile_pics_count,bio_length,age,relationship_intent,last_active_hour,emoji_usage_rate,is_success";
        
        String[] attrNames = keepAttributes.split(",");
        StringBuilder indices = new StringBuilder();
        for (String name : attrNames) {
            Attribute attr = rawData.attribute(name.trim());
            if (attr != null) indices.append(attr.index() + 1).append(",");
        }
        
        remove.setAttributeIndices(indices.substring(0, indices.length() - 1));
        remove.setInvertSelection(true);
        remove.setInputFormat(rawData);
        Instances data = Filter.useFilter(rawData, remove);

        // Set class to is_success (last attribute)
        data.setClassIndex(data.numAttributes() - 1);

        ReplaceMissingValues replace = new ReplaceMissingValues();
        replace.setInputFormat(data);
        data = Filter.useFilter(data, replace);

        return data;
    }

    public static Instances applySMOTE(Instances data) throws Exception {
        System.out.println("\nApplying SMOTE to balance classes...");
        if (data.classIndex() < 0) data.setClassIndex(data.numAttributes() - 1);

        SMOTE smote = new SMOTE();
        smote.setPercentage(100);
        smote.setInputFormat(data);
        Instances balancedData = Filter.useFilter(data, smote);
        System.out.println("New Dataset Size after SMOTE: " + balancedData.numInstances());
        return balancedData;
    }
}
