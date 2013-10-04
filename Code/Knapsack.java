import java.util.*;
public class Knapsack {
    private Set<Integer> sackResult;
    private double sackValue;
    
    private int getSackWeight(Map<Integer,Integer> weights,Set<Integer> sack) {
        int sum=0;
        for(Integer index:sack) {
            sum+=weights.get(index);
        }
        return sum;
    }
    
    public Knapsack(Map<Integer,Double> values,Map<Integer,Integer> weights,int capacity) {
        SortedMap<Double,Set<Integer>> adjustedWeights=new TreeMap<Double,Set<Integer>>();
        for(Integer index:weights.keySet()) {
            int weight=weights.get(index);
            double value=values.get(index);
            double adjustedValue=value/weight;
            if(adjustedWeights.get(adjustedValue)==null) {
                Set<Integer> tempSet=new HashSet<Integer>();
                adjustedWeights.put(adjustedValue,tempSet);
            }
            adjustedWeights.get(adjustedValue).add(index);
        }
        Set<Integer> sack=new HashSet<Integer>();
        double total=0;
        while(!adjustedWeights.isEmpty()&&getSackWeight(weights,sack)<capacity) {
            Set<Integer> indices=adjustedWeights.get(adjustedWeights.lastKey());
            for(Integer index:indices) {
                sack.add(index);
                int sackWeight=getSackWeight(weights,sack);
                if(sackWeight>capacity) {
                    sack.remove(index);
                } else {
                    total+=values.get(index);
                }
                if(sackWeight==capacity) {
                    break;
                }
            }
            adjustedWeights.remove(adjustedWeights.lastKey());
        }
        sackResult=sack;
        sackValue=total;
    }
    
    public Set<Integer> getSack() {
        return sackResult;
    }
    
    public double getSackValue() {
        return sackValue;
    }
}

