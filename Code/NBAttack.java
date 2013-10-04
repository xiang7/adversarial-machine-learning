import java.math.MathContext;
import java.math.BigDecimal;
import java.util.*;
public class NBAttack {
    private int bestClass;
    private double bestValue;
    private Set<Integer> bestPoint;
    private Map<Integer,Integer> wordLengths;
    private Map<Integer,Map<Integer,Double> > densities;
    private Map<Integer,Double> classProbabilities;
    private MathContext mc;
    private BigDecimal getValue(BigDecimal a,BigDecimal b,int i,int y) {
        BigDecimal tempA=a.multiply(new BigDecimal(densities.get(1).get(i)));
        BigDecimal tempB=b.multiply(new BigDecimal(densities.get(-1).get(i)));
        BigDecimal numerator=tempA.subtract(tempB).subtract(a.subtract(b));
        BigDecimal denominator=new BigDecimal(wordLengths.get(i)+1);
        BigDecimal retVal=numerator.divide(denominator,mc);
        return retVal.multiply(new BigDecimal(y));
    }
    private void setPoint() {
Random rand=new Random();
        for(int y=-1;y<=1;y+=2) {
            Map<Integer,Integer> weights=new HashMap<Integer,Integer>();
            Map<Integer,Double> values=new HashMap<Integer,Double>();
            for(Integer index:wordLengths.keySet()) {
                double x=1;
                x*=densities.get(1).get(index)/(1-densities.get(1).get(index));
                x/=densities.get(-1).get(index)/(1-densities.get(-1).get(index));
                x=Math.pow(x,y);
                x=Math.log(x);
                //if(rand.nextInt(1000)==0) {
                  //  System.out.println("P( C=1 | x_{"+index+"}=1 ) = "+densities.get(1).get(index));
                    //System.out.println("P( C=-1 | x_{"+index+"}=1 ) = "+densities.get(-1).get(index));
                    //System.out.println("value is "+x+" when maximizing for class "+y);
//System.out.println();
  //              }
                values.put(index,x);
                weights.put(index,wordLengths.get(index)+1);
            }
            Knapsack knapsack=new Knapsack(values,weights,1201);
    //        System.out.println("sack value " +knapsack.getSackValue());
      //      System.out.println("sack chosen " +knapsack.getSack());
            if(y==-1||knapsack.getSackValue()>bestValue) {
                bestValue=knapsack.getSackValue();
                bestPoint=knapsack.getSack();
                bestClass=y;
            }
        }
    }
    // CREATE attack, give a map representing word lengths, density values mapping classLabel to map of feature index to probability, map from class to probability of that class
    public NBAttack(Map<Integer,Integer> wordLengths,Map<Integer,Map<Integer,Double> > densities,Map<Integer,Double> classProbabilities) {
        this.wordLengths=wordLengths;
        this.densities=densities;
        this.classProbabilities=classProbabilities;
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
    public Set<String> attackTweet(Map<Integer,String> words) {
        Set<String> tweet=new HashSet<String>();
        for(Integer index:bestPoint) {
            tweet.add(words.get(index));
        }
        return tweet;
    }
}


