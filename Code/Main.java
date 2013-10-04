import java.util.*;
public class Main {
    public static void main(String[] args) {
        Attack a;
        NBAttack b;
        Map<Integer,Map<Integer,Double> > densities=new HashMap<Integer,Map<Integer,Double> >();
        Map<Integer,Integer> wordLengths=new HashMap<Integer,Integer>();
        Map<Integer,Double> classProbabilities=new HashMap<Integer,Double>();
        classProbabilities.put(-1,0.55);
        classProbabilities.put(1,0.45);
        Random rand=new Random();
        for(int j=0;j<20000;j++) {
            wordLengths.put(j,rand.nextInt(10)+3);
        }
        for(int i=-1;i<=1;i+=2) {
            Map<Integer,Double> innerDensities=new HashMap<Integer,Double>();
            for(int j=0;j<20000;j++) {
                if((j<10000&&i==-1)||(j>=10000&&i==1)) {
                    innerDensities.put(j,rand.nextDouble()/2);
                } else {
                    innerDensities.put(j,rand.nextDouble()/2+0.5);
                }
            }
            densities.put(i,innerDensities);
        }
        NBAttack atk=new NBAttack(wordLengths,densities,classProbabilities);
        Set<Integer> point=atk.attackPoint();
        int label=atk.attackLabel();
        System.out.println(label+" "+point);
for(Integer x:point) {
System.out.println(densities.get(-1).get(x)+" " +densities.get(1).get(x)+" "+wordLengths.get(x));
}
    }
}
