import java.util.*;
import java.io.*;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
public class NBCreateAttack{

	//Election: 1200, {6,30,60,300,600,1200}
	//Egyptian: 3000, {6,30,60,300,600,1200}
	private static double[] start_points=new double[] {0.25,0.5,0.75};
	private static double start_point=0.25;
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{

		for(int i=1;i<2;i++) {
			start_point=start_points[i];
			//target create
		run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				true,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+"CREATEACCTEST"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				1,
				false
				);
		}
	}
	
	/**
	 * Run the replicate experiment
	 * @param baseNum - the number of tweets to start with
	 * @param trainFile - the training files
	 * @param basePctg - the percentage of tweets to use in each training file for base case
	 * @param repPctg - the percentage of tweets to use in each training file when adding tweets
	 * @param flip - whether to flip the label or not when adding tweets
	 * @param seen - whether add seen data or unseen data into the training set
	 * @param testFile - the test file
	 * @param outFile - the file to write the output
	 * @param num - number of iterations to run
	 * @param maxNum - the number of tweets to stop
	 * @throws Exception
	 */
	public static void run(int baseNum,File[] trainFile,double[] basePctg,boolean targeted,File testFile,File accOutFile,int num,boolean continued) throws Exception
	{
		System.out.println(accOutFile.getAbsolutePath());
		Map<Integer,Integer> wordLength=CreateAttack.getWordLength(new File(ConstantClass.FeatureFile));
		Map<Integer,String> wordString= getWordString(new File(ConstantClass.FeatureFile));
		//read in all the training vectors
		BufferedWriter accOut=new BufferedWriter(new FileWriter(accOutFile,true));
		List<List<LabeledFeatureVector>> l=new Vector<List<LabeledFeatureVector>>();
		//read vectors in from each train file
		int total=0;
		//read vectors in from each train file
		for(int i=0;i<trainFile.length;i++)
			{
			List<LabeledFeatureVector> tempL=Tool.readFeaVec(trainFile[i]);
			l.add(tempL);
			total+=tempL.size();
			}
		//read vectors from the test file
		List<LabeledFeatureVector> t=Tool.readFeaVec(testFile);
		LabeledFeatureVector[] te=t.toArray(new LabeledFeatureVector[t.size()]);
		//train and test
		for(int i=0;i<num;i++)
		{
			System.out.println(i+"th run");
			//generate baseNum of tweets
			List<List<LabeledFeatureVector>> baseNumTwt=new Vector<List<LabeledFeatureVector>>();
			List<LabeledFeatureVector> train=new Vector<LabeledFeatureVector>();
			//add a max feature into the last sample
			LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			train.add(tempL);
			//get the base training data
			for(int j=0;j<trainFile.length;j++)
			{
				baseNumTwt.add(Tool.getRandVec(l.get(j), (int)(baseNum*basePctg[j]),false));
				train.addAll(baseNumTwt.get(j));
			}
			
			double acc=1.0;
			String ct="";
			for(;acc>0.5&&train.size()-start_point*ConstantClass.DataSetSize<31000;)
			{
				NBAttack attack=null;
				LabeledFeatureVector[] tr=train.toArray(new LabeledFeatureVector[train.size()]);
				MyNaiveBayesMultinomial nb=TrainTestOut.trainBayes(tr);
				acc=TrainTestOut.testBayes(nb, te);
				Map<Integer,Map<Integer,Double>> density=getDensityMap(nb.toString(),wordString);
				Map<Integer,Double> classProb=getClassProbMap(nb.getClassProb());
				attack=new NBAttack(wordLength, density,classProb);
				ct=attack.attackLabel()+"\t"+attack.attackTweet(wordString);		//created tweet
				System.out.println(ct);
				Set<Integer> point=attack.attackPoint();
				LabeledFeatureVector attackVec=point2Vec(attack.attackLabel(), point);
				//for(Integer x:point) {c
				//	System.out.println(density.get(-1).get(x)+" " +density.get(1).get(x)+" "+wordLength.get(x));
				//	}
				accOut.write(acc+"\t");
				accOut.flush();
				System.out.println("Trained on "+(train.size()-1)+" tweets ( "+(double)train.size()/(double)total+" of all the data ) with acc: "+acc);
				//add data ten times to the training set
				for(int j=0;j<10;j++)
					{
						train.add(attackVec);
					}
				System.gc();
			}
			accOut.write("\n");
			accOut.flush();
	}
		accOut.close();
}
	
	
	
	public static LabeledFeatureVector point2Vec(double classLabel,Set<Integer> set)
	{
		LabeledFeatureVector l=new LabeledFeatureVector(classLabel,set.size());
		TreeSet<Integer> s=new TreeSet<Integer>();
		Iterator<Integer> it=set.iterator();
		for(int i=0;i<set.size();i++)
			s.add(it.next());
		int[] idx=new int[set.size()];
		double[] val=new double[set.size()];
		Iterator<Integer> sorted_it=s.iterator();
		for(int i=0;i<set.size();i++)
		{
			idx[i]=sorted_it.next();
			val[i]=1.0;
		}
		l.setFeatures(idx, val);
		return l;
	}
	
	public static Map<Integer,Integer> getWordLength(File feaFile) throws Exception
	{
		Hashtable<Integer,Integer> table = new Hashtable<Integer, Integer>(); 
		BufferedReader in = new BufferedReader(new FileReader(feaFile));
		String temp="";
		while((temp=in.readLine())!=null)
		{
			String[] s=temp.split(ConstantClass.delimiter);
			table.put(Integer.valueOf(s[1]), s[0].length());
		}
		in.close();
		return table;
	}
	
	public static Map<Integer,String> getWordString(File feaFile) throws Exception
	{
		Hashtable<Integer,String> table = new Hashtable<Integer, String>(); 
		BufferedReader in = new BufferedReader(new FileReader(feaFile));
		String temp="";
		while((temp=in.readLine())!=null)
		{
			String[] s=temp.split(ConstantClass.delimiter);
			table.put(Integer.valueOf(s[1]), s[0]);
		}
		in.close();
		return table;
	}
	
	//which is 1 which is -1?
	public static Map<Integer,Double> getClassProbMap(double[] classProb)
	{
		Map<Integer,Double> m=new Hashtable<Integer, Double>();
		m.put(1, classProb[0]);
		m.put(-1, classProb[1]);
		return m;
	}
	
	//which is the starting point? which is 1 which is -1
	public static Map<Integer,Map<Integer,Double>> getDensityMap(String total,Map<Integer,String> wordString)
	{
		int start=total.indexOf("0\t");
		total=total.substring(start);
		String[] totalSplit=total.split("\n");
		Map<Integer,Map<Integer,Double>> m=new Hashtable<Integer, Map<Integer,Double>>();
		Map<Integer,Double> pos=new Hashtable<Integer, Double>();
		Map<Integer,Double> neg=new Hashtable<Integer, Double>();
		for(int i=1;i<totalSplit.length;i++)
		{
			String value[]=totalSplit[i].split("\t");
			//if((Double.valueOf(value[2])-Double.valueOf(value[1]))/Double.valueOf(value[2])>0.9)
			//	System.out.println(wordString.get(Integer.valueOf(value[0])));
			pos.put(i, Double.valueOf(value[1]));
			neg.put(i, Double.valueOf(value[2]));
			}
		m.put(1,pos);
		m.put(-1,neg);
		return m;
	}
	
	
}
