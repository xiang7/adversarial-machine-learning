import java.util.*;
public class Attack {
    private int bestClass;
    private double bestValue;
    private Set<Integer> bestPoint;
    private Map<Integer,Integer> wordLengths;
    private Map<Integer,Double> svmVector;
    private Map<Integer,List<Set<Integer> > > testSet;
    private double createValue(int index,int classLabel) {
        return svmVector.get(index)*classLabel;
    }
    private double targettedCreateValue(int index,int classLabel) {
        int testSum=0;
        for(Integer testClass:testSet.keySet()) {
            List<Set<Integer> > classPoints=testSet.get(testClass);
            for(Set<Integer> point:classPoints) {
                if(point.contains(index)) {
                    testSum+=testClass*classLabel;
                }
            }
        }
        double classifierValue=createValue(index,classLabel);
        if(classifierValue<0&&testSum<0)
        	return -1;
        return classifierValue*testSum;
    }
    private void setPoint() {
    	this.bestValue=-1;
        Map<Integer,Integer> wordWeights=new HashMap<Integer,Integer>();
        for(Integer index:wordLengths.keySet()) {
            wordWeights.put(index,wordLengths.get(index)+1);
        }
        for(int classLabel=-1;classLabel<=1;classLabel+=2) {
            Map<Integer,Double> wordValues=new HashMap<Integer,Double>();
            for(Integer index:svmVector.keySet()) {
                double value=0;
                if(testSet==null) {
                    value=createValue(index,classLabel);
                } else {
                    value=targettedCreateValue(index,classLabel);
                }                
                wordValues.put(index,value);
            }
            Knapsack sack=new Knapsack(wordValues,wordWeights,141);
            if(sack.getSackValue()>bestValue) {
                bestPoint=sack.getSack();
                bestValue=sack.getSackValue();
                bestClass=classLabel;
                //System.out.println(bestValue);
            }
        }
    }
    // CREATE attack, give a map representing word lengths and an svm weight vector
    public Attack(Map<Integer,Integer> wordLengths,Map<Integer,Double> svmVector) {
        this.wordLengths=wordLengths;
        this.svmVector=svmVector;
        this.testSet=null;
        setPoint();
    }
    // TARGETTED-CREATE attack, give a map representing word lengths, an svm weight vector, and a map for the testset
    // The testset map has a key of the class label with a value that is a list of datapoints
    // Datapoints are sets of integers where the integers represent indices of the words
    public Attack(Map<Integer,Integer> wordLengths,Map<Integer,Double> svmVector,Map<Integer,List<Set<Integer> > > testSet) {
        this.wordLengths=wordLengths;
        this.svmVector=svmVector;
        this.testSet=testSet;
        setPoint();
    }
    // get the attack point itself which is a set of indices where the value equals 1
    public Set<Integer> attackPoint() {
        return bestPoint;
    }
    // get the label that would be placed on an attack point
    public int attackLabel() {
        return -1*bestClass;
    }
    // get an attack tweet if you have a mapping from indices to words
    public String attackTweet(Map<Integer,String> words) {
        Set<String> tweet=new HashSet<String>();
        for(Integer index:bestPoint) {
            tweet.add(words.get(index));
        }
        StringBuilder sb=new StringBuilder();
        for (String t : tweet)
        	sb.append(t+" ");
        return new String(sb);
    }
}

