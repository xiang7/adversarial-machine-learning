import java.util.*;
import java.io.*;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
public class IncrementalTrain{

	public static void main(String[] args) throws Exception{
			//create
			//System.loadLibrary("/home/xiang7/Documents/Purdue/Adversarial Machine Learning/project/CCS2013/JNI_SVM-light-6.01/lib/svmlight");
			run(	new File[] {new File(ConstantClass.folder+"pos2Vec"),new File(ConstantClass.folder+"neg2Vec")},
					new File(ConstantClass.folder+"attackVec"),
					new File(ConstantClass.TestVec),
					new File(ConstantClass.folder+"CREATEACC2"+ConstantClass.currClassifierName)
					);
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
	public static void run(File[] trainFile,File attackVecFile,File testFile,File accOutFile) throws Exception
	{
		//read in all the training vectors
		BufferedWriter accOut=new BufferedWriter(new FileWriter(accOutFile,true));
		List<LabeledFeatureVector> train=new Vector<LabeledFeatureVector>();
		List<LabeledFeatureVector> attackVec=Tool.readFeaVec(attackVecFile);
		//read vectors in from each train file
		int total=0;
		//read vectors in from each train file
		for(int i=0;i<trainFile.length;i++)
		{
			List<LabeledFeatureVector> tempL=Tool.readFeaVec(trainFile[i]);
			train.addAll(tempL);
			total+=tempL.size();
		}
		//read vectors from the test file
		List<LabeledFeatureVector> t=Tool.readFeaVec(testFile);
		LabeledFeatureVector[] te=t.toArray(new LabeledFeatureVector[t.size()]);
		//train and test 

		//add a max feature into the last sample
		LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
		train.add(tempL);

		double acc=1.0;//accuracy
		int curr_attack=0;
		for(;acc>0.5;)
		{
			LabeledFeatureVector[] tr=train.toArray(new LabeledFeatureVector[train.size()]);
			SVMLightModel m=TrainTestOut.train(tr);
			acc=TrainTestOut.test(m,te);
			//add data ten times to the training set
			for(int j=0;j<10;j++)
			{
				train.add(attackVec.get(curr_attack));
			}
			curr_attack++;
			accOut.write(acc+"\n");
			accOut.flush();
			System.out.println("Trained on "+(train.size()-1)+" tweets ( "+(double)train.size()/(double)total+" of all the data ) with acc: "+acc);
			System.gc();
		}
		accOut.flush();
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

public static Map<Integer,Double> getSVMVector(SVMLightModel m)
{
	Map<Integer,Double> map=new Hashtable<Integer, Double>();
	double[] d = m.getLinearWeights();
	for(int i=1;i<d.length;i++)
		map.put(i, d[i]);
	return map;
}

public static Map<Integer,List<Set<Integer>>> getTestSet(List<LabeledFeatureVector> test)
{
	List<Set<Integer>> pos=new Vector<Set<Integer>>();
	List<Set<Integer>> neg=new Vector<Set<Integer>>();
	for(LabeledFeatureVector l:test)
	{
		if(l.getLabel()>0)//pos
		{
			Set<Integer> set=new TreeSet<Integer>();
			for(int i=0;i<l.size();i++)
				set.add(l.getDimAt(i));
			pos.add(set);
		}
		else {
			Set<Integer> set=new TreeSet<Integer>();
			for(int i=0;i<l.size();i++)
				set.add(l.getDimAt(i));
			neg.add(set);
		}
	}

	Map<Integer,List<Set<Integer>>> map=new Hashtable<Integer, List<Set<Integer>>>();
	map.put(1, pos);
	map.put(-1, neg);
	return map;
}
}
