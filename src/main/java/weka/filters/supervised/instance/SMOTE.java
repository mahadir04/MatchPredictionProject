package weka.filters.supervised.instance;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.util.*;

/**
 * Functional SMOTE implementation for class imbalance.
 */
public class SMOTE extends Filter implements SupervisedFilter, OptionHandler {

    private int m_Percentage = 200;
    private int m_NearestNeighbors = 5;

    public String globalInfo() {
        return "Resamples a dataset by applying the Synthetic Minority Over-sampling TEchnique (SMOTE).";
    }

    public void setPercentage(int value) { m_Percentage = value; }
    public int getPercentage() { return m_Percentage; }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.setMinimumNumberInstances(0);
        return result;
    }

    @Override
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        setOutputFormat(instanceInfo);
        return true;
    }

    @Override
    public boolean input(Instance instance) throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_FirstBatchDone) {
            push((Instance) instance.copy());
            return true;
        } else {
            bufferInput(instance);
            return false;
        }
    }

    @Override
    public boolean batchFinished() throws Exception {
        if (getInputFormat() == null) throw new IllegalStateException("No input instance format defined");

        Instances input = getInputFormat();
        
        // Separate instances by class
        Map<Double, List<Instance>> classMap = new HashMap<>();
        for (int i = 0; i < input.numInstances(); i++) {
            double classValue = input.instance(i).classValue();
            classMap.computeIfAbsent(classValue, k -> new ArrayList<>()).add(input.instance(i));
        }

        // Push all original instances to output queue
        for (int i = 0; i < input.numInstances(); i++) {
            push((Instance) input.instance(i).copy());
        }

        // Find minority class
        double minorityClassValue = -1;
        int minCount = Integer.MAX_VALUE;
        for (Map.Entry<Double, List<Instance>> entry : classMap.entrySet()) {
            if (entry.getValue().size() < minCount) {
                minCount = entry.getValue().size();
                minorityClassValue = entry.getKey();
            }
        }

        if (minorityClassValue != -1 && classMap.get(minorityClassValue).size() > 1) {
            List<Instance> minorityInstances = classMap.get(minorityClassValue);
            int numToCreate = (int) (minorityInstances.size() * (m_Percentage / 100.0));

            Random random = new Random(1);
            for (int i = 0; i < numToCreate; i++) {
                Instance sample = minorityInstances.get(random.nextInt(minorityInstances.size()));
                Instance neighbor = minorityInstances.get(random.nextInt(minorityInstances.size()));
                
                double[] values = new double[input.numAttributes()];
                for (int j = 0; j < input.numAttributes(); j++) {
                    if (input.attribute(j).isNumeric()) {
                        double diff = neighbor.value(j) - sample.value(j);
                        values[j] = sample.value(j) + random.nextDouble() * diff;
                    } else {
                        values[j] = random.nextBoolean() ? sample.value(j) : neighbor.value(j);
                    }
                }
                push(new DenseInstance(1.0, values));
            }
        }

        flushInput();
        m_FirstBatchDone = true;
        return (numPendingOutput() > 0);
    }

    @Override
    public Enumeration<Option> listOptions() { return null; }
    @Override
    public void setOptions(String[] options) throws Exception {}
    @Override
    public String[] getOptions() { return new String[0]; }
}
