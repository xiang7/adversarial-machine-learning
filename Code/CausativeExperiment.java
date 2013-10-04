import java.util.*;
import java.io.*;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
public class CausativeExperiment {

	//Election: 1200, {6,30,60,300,600,1200}
	//Egyptian: 3000, {6,30,60,300,600,1200}
	private static int k_max=5;
	private static int stepSize=200;
	private static double start_point=0.5;
	private static int end_point=ConstantClass.DataSetSize;
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		/**
		 * replicate flip attack
		 */
		for(int i=1;i<2;i++) {
			if(i==1)
				start_point=0.5;
			else if(i==2)
				start_point=0.75;
			end_point=(int)(start_point*ConstantClass.DataSetSize)+(int)(ConstantClass.DataSetSize*0.5)+stepSize;
		/*run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0.5,0.5},
				true,
				true,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//ReplicateFlip"),
				5,
				(int)(end_point));
		//System.exit(0);
		/**
		 * replicate non-flip attack
		 *//*
		run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0,1},
				false,
				true,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//ReplicateNegative"),
				5,
				(int)(end_point));*/
		/**
		 * new flip attack
		 *//*
		run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0.5,0.5},
				true,
				false,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+"SCORE//LABELCHANGEACC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"SCORE//LABELCHANGEWC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"SCORE//LABELCHANGEAS"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				1,
				(int)(end_point));*/
		/**
		 * new non-flip attack
		 */
		run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0,1},
				false,
				false,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+"SCORE//REPEATACC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"SCORE//REPEATWC"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				new File(ConstantClass.folder+"SCORE//REPEATAS"+ConstantClass.currClassifierName+String.valueOf((int)(start_point*100))),
				1,
				(int)(end_point));
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
	public static void run(int baseNum,File[] trainFile,double[] basePctg,double[] repPctg,boolean flip,boolean seen,File testFile,File accOutFile,File wcOutFile,File asOutFile,int num, int maxNum) throws Exception
	{
		double[] w_previous=null;
		BufferedWriter accOut=new BufferedWriter(new FileWriter(accOutFile,true));
		BufferedWriter wcOut=new BufferedWriter(new FileWriter(wcOutFile,true));
		BufferedWriter asOut=new BufferedWriter(new FileWriter(asOutFile,true));
		//read in all the training vectors
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
		//train and test
		for(int k=baseNum;k<maxNum;k+=stepSize)
			accOut.write(k+"\t");
		accOut.write("\n");
		for(int i=0;i<num;i++)
		{
			System.out.println(i+"th run");
			//generate baseNum of tweets
			List<List<LabeledFeatureVector>> baseNumTwt=new Vector<List<LabeledFeatureVector>>();
			List<LabeledFeatureVector> train=new Vector<LabeledFeatureVector>();
			List<List<LabeledFeatureVector>> lCopy=new Vector<List<LabeledFeatureVector>>();
			if(!seen)
				for(int j=0;j<trainFile.length;j++)
					lCopy.add(new Vector<LabeledFeatureVector>(l.get(j)));
			//add a max feature into the last sample
			LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			train.add(tempL);
			//get the base training data
			for(int j=0;j<trainFile.length;j++)
			{
				if(seen)
					baseNumTwt.add(Tool.getRandVec(l.get(j), (int)(baseNum*basePctg[j]),false));
				else
					baseNumTwt.add(Tool.getRandVecRemove(lCopy.get(j), (int)(baseNum*basePctg[j])));
				train.addAll(baseNumTwt.get(j));
			}
			for(int k=baseNum;k<maxNum;k+=stepSize)
			{
				LabeledFeatureVector[] te=t.toArray(new LabeledFeatureVector[t.size()]);
				LabeledFeatureVector[] tr=train.toArray(new LabeledFeatureVector[train.size()]);
				SVMLightModel m=null;
				MyNaiveBayesMultinomial nb=null;
				AdaBoost ab=null;
				if(ConstantClass.currClassifier==ConstantClass.ADABOOST)
					ab=new AdaBoost(k_max,train); 
				double acc=0.0;
				double wc=0.0;
				double as=0.0;
				if(ConstantClass.currClassifier==ConstantClass.SVM)
					{
					m=TrainTestOut.train(tr);
					m.writeModelToFile("I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\CCS2013\\Data\\Egyptian\\Figures\\Exploratory\\UnseenExploratory\\model");
					acc=TrainTestOut.test(m,te);
					wc=Tool.weight_change_ratio(w_previous, m.getLinearWeights());		//weight change ratio
					as=Tool.angular_diff(w_previous, m.getLinearWeights());				//angular shift
					w_previous=m.getLinearWeights();
					}
				else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
					{
					nb=TrainTestOut.trainBayes(tr);
					acc=TrainTestOut.testBayes(nb,te);
					}
				else if(ConstantClass.currClassifier==ConstantClass.ADABOOST)
					{
					ab.train();
					acc=ab.test(te);
					}
				accOut.write(acc+"\t");
				wcOut.write(wc+"\t");
				asOut.write(as+"\t");
				accOut.flush();
				wcOut.flush();
				asOut.flush();
				System.out.println("Trained on "+(train.size()-1)+" tweets ( "+(double)train.size()/(double)total+" of all the data ) with acc: "+acc);
				//add data to the training set
				if(seen) {
				for(int j=0;j<trainFile.length;j++)
					if(!flip)
						train.addAll(Tool.getRandVec(baseNumTwt.get(j), (int)(stepSize*repPctg[j]),true));
					else
						train.addAll(flip(Tool.getRandVec(baseNumTwt.get(j), (int)(stepSize*repPctg[j]),true)));
				}
				else {
					for(int j=0;j<trainFile.length;j++)
						if(!flip)
							train.addAll(Tool.getRandVec(lCopy.get(j), (int)(stepSize*repPctg[j]),true));
						else
							train.addAll(flip(Tool.getRandVec(lCopy.get(j), (int)(stepSize*repPctg[j]),true)));
				}
				System.gc();
			}
			accOut.write("\n");
			wcOut.write("\n");
			asOut.write("\n");
			accOut.flush();
			wcOut.flush();
			asOut.flush();
	}
		accOut.close();
		wcOut.close();
		asOut.close();

}
	
	/**
	 * Flip the labels of vectors in this list
	 * @param l
	 */
	public static List<LabeledFeatureVector> flip(List<LabeledFeatureVector> l)
	{
		List<LabeledFeatureVector> temp=new Vector<LabeledFeatureVector>();
		for(LabeledFeatureVector lTemp : l)
		{
			int dims[]=new int[lTemp.size()];
			double vals[]=new double[lTemp.size()];
			for(int i=0;i<lTemp.size();i++)
			{
				dims[i]=lTemp.getDimAt(i);
				vals[i]=lTemp.getValueAt(i);
			}
			LabeledFeatureVector f=new LabeledFeatureVector(-(lTemp.getLabel()), dims,vals);
			temp.add(f);
		}
		return temp;
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
		return table;
		
	}
}
