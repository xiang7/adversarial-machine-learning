import java.util.*;
import java.io.*;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
public class CreateAttackPoints{

	public static void main(String[] args) throws Exception{
			//create
			run(	new File[] {new File(ConstantClass.folder+"pos1Vec"),new File(ConstantClass.folder+"neg1Vec")},
					false,
					new File(ConstantClass.TestVec),
					new File(ConstantClass.folder+"CREATEACC1"+ConstantClass.currClassifierName),
					new File(ConstantClass.folder+"CREATECT1POS"+ConstantClass.currClassifierName),
					new File(ConstantClass.folder+"CREATECT1NEG"+ConstantClass.currClassifierName),
					new File(ConstantClass.folder+"CREATECV1"+ConstantClass.currClassifierName) //to store the created attack vectors, for continued training/attack
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
	public static void run(File[] trainFile,boolean targeted,File testFile,File accOutFile,File ctOutFilePos,File ctOutFileNeg, File cvOutFile) throws Exception
	{
		//get the length of each word in the feature space <Fea ID, Length of word>
		Map<Integer,Integer> wordLength=CreateAttack.getWordLength(new File(ConstantClass.FeatureFile));
		//get the string of each word in the feature space <Fea ID, string>
		Map<Integer,String> wordString= getWordString(new File(ConstantClass.FeatureFile));
		double w_previous[]=null;
		//read in all the training vectors
		BufferedWriter accOut=new BufferedWriter(new FileWriter(accOutFile,true));
		BufferedWriter ctOutPos=new BufferedWriter(new FileWriter(ctOutFilePos,true));
		BufferedWriter ctOutNeg=new BufferedWriter(new FileWriter(ctOutFileNeg,true));
		BufferedWriter cvOut=new BufferedWriter(new FileWriter(cvOutFile,true));
		List<LabeledFeatureVector> train=new Vector<LabeledFeatureVector>();
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
		Map<Integer,List<Set<Integer>>> testSet=null;
		if (targeted)
			testSet=getTestSet(t);
		LabeledFeatureVector[] te=t.toArray(new LabeledFeatureVector[t.size()]);
		//train and test

		//add a max feature into the last sample
		LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {wordString.size()},new double[] {1.0});
		train.add(tempL);

		double acc=1.0;//accuracy
		String ct="";//created tweet
		int id=0;
		for(;acc>0.5;)
		{
			Attack attack=null;
			LabeledFeatureVector[] tr=train.toArray(new LabeledFeatureVector[train.size()]);
			SVMLightModel m=TrainTestOut.train(tr);
			acc=TrainTestOut.test(m,te);
			if(!targeted)//create attack
				attack=new Attack(wordLength, getSVMVector(m));
			else
				attack=new Attack(wordLength, getSVMVector(m), testSet);
			String s_id=String.valueOf(id);
			while(s_id.length()!=5)
				s_id="0"+s_id;
			ct=s_id+"\t"+attack.attackTweet(wordString);		//created tweet
			id++;
			w_previous=m.getLinearWeights();
			LabeledFeatureVector attackVec=point2Vec(attack.attackLabel(), attack.attackPoint());
			cvOut.write("0\t"+attackVec.toString());
			//add data ten times to the training set
			for(int j=0;j<10;j++)
			{
				train.add(attackVec);
			}
			accOut.write(acc+"\n");
			if(attack.attackLabel()>0)
				ctOutPos.write(ct+"\n");
			else
				ctOutNeg.write(ct+"\n");
			accOut.flush();
			cvOut.flush();
			ctOutPos.flush();
			ctOutNeg.flush();
			System.out.println("Trained on "+(train.size()-1)+" tweets ( "+(double)train.size()/(double)total+" of all the data ) with acc: "+acc);
			System.gc();
		}
		accOut.flush();
		cvOut.flush();
		ctOutPos.flush();
		ctOutNeg.flush();
		accOut.close();
		ctOutPos.close();
		ctOutNeg.close();
		cvOut.close();
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
