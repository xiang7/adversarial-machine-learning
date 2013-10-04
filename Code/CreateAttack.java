import java.util.*;
import java.io.*;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
public class CreateAttack{

	//Election: 1200, {6,30,60,300,600,1200}
	//Egyptian: 3000, {6,30,60,300,600,1200}
	private static double[] start_points=new double[] {1.0};
	private static double start_point=0.25;
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		/**
		 * replicate flip attack
		 */
		for(int i=0;i<1;i++) {
			start_point=start_points[i];
			//create
			/*
			run((int)(start_point*ConstantClass.DataSetSize),
					new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
					new double[] {0.5,0.5},
					false,
					new File(ConstantClass.TestVec),
					new File(ConstantClass.folder+"CREATEACC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATEWC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATEAS"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATECT"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATEVEC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					1,
					true
					);*/
		}
		for(int i=0;i<2;i++) {
			start_point=start_points[i];
			//create
			
			run((int)(start_point*ConstantClass.DataSetSize),
					new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
					new double[] {0.5,0.5},
					false,
					new File(ConstantClass.TestVec),
					new File(ConstantClass.folder+"CREATEACC1"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATEWC1"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATEAS1"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATECT1"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					new File(ConstantClass.folder+"CREATEVEC1"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
					1,
					true
					);
		}
		System.exit(0);
		for(int i=1;i<2;i++) {
			start_point=start_points[i];
			//target create
		run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				true,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+"TARGETCREATEACC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"TARGETCREATEWC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"TARGETCREATEAS"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"TARGETCREATECT"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"TARGETCREATEVEC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
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
	public static void run(int baseNum,File[] trainFile,double[] basePctg,boolean targeted,File testFile,File accOutFile,File wcOutFile,File asOutFile,File ctOutFile,File vecOutFile,int num,boolean continued) throws Exception
	{
		//get the length of each word in the feature space <Fea ID, Length of word>
		Map<Integer,Integer> wordLength=CreateAttack.getWordLength(new File(ConstantClass.FeatureFile));
		//get the string of each word in the feature space <Fea ID, string>
		Map<Integer,String> wordString= getWordString(new File(ConstantClass.FeatureFile));
		double w_previous[]=null;
		List<LabeledFeatureVector> existingVec=null;
		//Enable continued training, if continued, read from vecOutFile and continue attacking 
		if(continued)
			existingVec=Tool.readFeaVec(vecOutFile);
		//read in all the training vectors
		BufferedWriter accOut=new BufferedWriter(new FileWriter(accOutFile,true));
		BufferedWriter wcOut=new BufferedWriter(new FileWriter(wcOutFile,true));
		BufferedWriter asOut=new BufferedWriter(new FileWriter(asOutFile,true));
		BufferedWriter ctOut=new BufferedWriter(new FileWriter(ctOutFile,true));
		BufferedWriter vecOut=new BufferedWriter(new FileWriter(vecOutFile,true));//append to the end
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
		Map<Integer,List<Set<Integer>>> testSet=getTestSet(t);
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
			if(continued)
				train.addAll(existingVec);
			else {
				//get the base training data
				for(int j=0;j<trainFile.length;j++)
				{
					baseNumTwt.add(Tool.getRandVec(l.get(j), (int)(baseNum*basePctg[j]),false));
					train.addAll(baseNumTwt.get(j));
				}
				//if first time, write the base tweets into the file
				for(LabeledFeatureVector tempTrain : train)
					vecOut.write("0\t"+tempTrain.toString());
			}
			
			double acc=1.0;//accuracy
			double as=0;//angular shift
			String ct="";//created tweet
			double wc=0;//weight change
			for(;acc>0.5&&train.size()-start_point*ConstantClass.DataSetSize<31000;)
			{
				Attack attack=null;
				LabeledFeatureVector[] tr=train.toArray(new LabeledFeatureVector[train.size()]);
				SVMLightModel m=TrainTestOut.train(tr);
				acc=TrainTestOut.test(m,te);
				if(!targeted)//create attack
					attack=new Attack(wordLength, getSVMVector(m));
				else
					attack=new Attack(wordLength, getSVMVector(m), testSet);
				ct=attack.attackLabel()+"\t"+attack.attackTweet(wordString);		//created tweet
				wc=Tool.weight_change_ratio(w_previous, m.getLinearWeights());		//weight change ratio
				as=Tool.angular_diff(w_previous, m.getLinearWeights());				//angular shift
				w_previous=m.getLinearWeights();
				LabeledFeatureVector attackVec=point2Vec(attack.attackLabel(), attack.attackPoint());
				//add data ten times to the training set
				for(int j=0;j<10;j++)
					{
						train.add(attackVec);
						vecOut.write("0\t"+attackVec.toString());
					}
				vecOut.write("\r\n");
				accOut.write(acc+"\t");
				ctOut.write(ct+"\r\n");
				wcOut.write(wc+"\t");
				asOut.write(as+"\t");
				accOut.flush();
				vecOut.flush();
				ctOut.flush();
				wcOut.flush();
				asOut.flush();
				System.out.println("Trained on "+(train.size()-1)+" tweets ( "+(double)train.size()/(double)total+" of all the data ) with acc: "+acc);
				System.gc();
			}
			accOut.write("\n");
			ctOut.write("\n");
			wcOut.write("\n");
			asOut.write("\n");
			vecOut.write("\r\n");
			accOut.flush();
			ctOut.flush();
			wcOut.flush();
			asOut.flush();
			vecOut.flush();
	}
		accOut.close();
		ctOut.close();
		wcOut.close();
		asOut.close();
		vecOut.close();
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
