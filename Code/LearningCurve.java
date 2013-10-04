import java.io.*;
import java.util.*;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import jnisvmlight.*;

public class LearningCurve {

	//Election: 1200, {6,30,60,300,600,1200}
	//Egyptian: 3000, {6,30,60,300,600,1200}
	private static int stepSize=3000;
	private static int[] startingPoints=new int[] {6,30,60,300,600,1200,1800};
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		run(new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new File(ConstantClass.TestVec),
				new double[] {0.5,0.5},
				new File(ConstantClass.folder+"\\1111LearningCurve"),5,ConstantClass.DataSetSize);

	}
	
	/**
	 * Run the learning curve experiment
	 * @param trainFile
	 * @param testFile
	 * @param pctg - percentage of tweets from each class
	 * @param outFile
	 * @param num - number of iterations
	 * @throws Exception
	 */
	public static void run(File[] trainFile,File testFile,double[] pctg,File outFile,int num,int maxNum) throws Exception
	{
		BufferedWriter out=new BufferedWriter(new FileWriter(outFile));
		List<List<LabeledFeatureVector>> l=new Vector<List<LabeledFeatureVector>>();
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
		Vector<Integer> points=new Vector<Integer>();
		points.add(0);
		for(int k=0;k<startingPoints.length;k++)
			{
			out.write(startingPoints[k]+"\t");
			points.add(startingPoints[k]);
			}
		for(int k=stepSize;k<maxNum;k+=stepSize)
			{
			out.write(k+"\t");
			points.add(k);
			}
		out.write("\n");
		for(int i=0;i<num;i++)
		{
			//add a max feature into the last sample
			LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			System.out.println(i+"th run");
			for(int m=1;m<points.size();m++)
			{
				List<LabeledFeatureVector> train=new Vector<LabeledFeatureVector>();
				train.add(tempL);
				int k=points.elementAt(m);
				for(int j=0;j<trainFile.length;j++)
					train.addAll(Tool.getRandVec(l.get(j), (int)((k)*pctg[j]),false));
				SVMLightModel model=null;
				MyNaiveBayesMultinomial nb=null;
				if(ConstantClass.currClassifier==ConstantClass.SVM)
					model=TrainTestOut.train(train.toArray(new LabeledFeatureVector[train.size()]));
				else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
					nb=TrainTestOut.trainBayes(train.toArray(new LabeledFeatureVector[train.size()]));
				//printModel(model,new File(ConstantClass.folder+"model"),new File(ConstantClass.FeatureFile));
				double acc=0;
				if(ConstantClass.currClassifier==ConstantClass.SVM)
					acc=TrainTestOut.test(model, t.toArray(new LabeledFeatureVector[t.size()]));
				else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
					acc=TrainTestOut.testBayes(nb, t.toArray(new LabeledFeatureVector[t.size()]));
				out.write(acc+"\t");
				out.flush();
				System.out.println("Trained on "+(train.size()-1)+" tweets ( "+(double)k/(double)total+" of all the data ) with acc: "+acc);
			}
			out.write("\r\n");
		}
		out.flush();
		out.close();
	}
	
	/**
	 * Print the model to a file. The model is specified by the features and weights.
	 * @param model
	 * @param outFile
	 * @param feaFile
	 * @throws Exception
	 */
public static void printModel(SVMLightModel model,File outFile,File feaFile) throws Exception
{
	BufferedReader in =new BufferedReader(new FileReader(feaFile));
	BufferedWriter out =new BufferedWriter(new FileWriter(outFile));
	Hashtable<Integer,String> map=new Hashtable<Integer, String>();
	String temp="";
	while((temp=in.readLine())!=null)
	{
		String split[]=temp.split(ConstantClass.delimiter);
		map.put(Integer.valueOf(split[1]), split[0]);
	}
	double[] weights=model.getLinearWeights();
	for(int i=1;i<weights.length;i++)
	{
		out.write(weights[i]+ConstantClass.delimiter+map.get(i)+"\n");
	}	
	in.close();
	out.flush();
	out.close();
	}
}
