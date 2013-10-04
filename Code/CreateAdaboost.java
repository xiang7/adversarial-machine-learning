import java.io.BufferedWriter;

import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Vector;
import jnisvmlight.LabeledFeatureVector;

public class CreateAdaboost {
	
	private static int stepSize=200;
	private static double start_point=0.5;
	private static double pctGood=0.5;
	private static int end_point=(int)(ConstantClass.DataSetSize*(start_point+0.5)+stepSize);
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{

		run(	5,
				(int)(start_point*ConstantClass.DataSetSize),
				new File("I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\CCS2013\\Data\\FINAL\\Egyptian\\AdaboostCreate\\CREATEVECSVM50"),
				new File(ConstantClass.TestVec),
				new File("I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\CCS2013\\Data\\FINAL\\Egyptian\\AdaboostCreate\\CREATEACC")
				);
		
	}
	
	/**
	 * run the create attack on adaboost
	 * @param numClassifiers - how many classifiers to use in the adaboost algorithm
	 * @param baseNum - 
	 * @param trainFile
	 * @param testFile
	 * @param outFile
	 * @param outAve
	 * @param outTimeFile
	 * @param num
	 * @param withReplacement
	 * @throws Exception
	 */
	public static void run(int numClassifiers,int baseNum,File trainFile,File testFile,File outFile) throws Exception
	{
		//read in all the training vectors
		BufferedWriter out=new BufferedWriter(new FileWriter(outFile,true));
		List<LabeledFeatureVector> l=new Vector<LabeledFeatureVector>();
		//read vectors in from each train file
		int total=0;
		//read vectors in from each train file
		l=Tool.readFeaVec(trainFile);
		total+=l.size();
		System.out.println("Total train size: "+total);
		//read vectors from the test file
		List<LabeledFeatureVector> t=Tool.readFeaVec(testFile);
		//test tweets
		LabeledFeatureVector[] te=t.toArray(new LabeledFeatureVector[t.size()]);
				
		//train and test
		List<LabeledFeatureVector> train=new Vector<LabeledFeatureVector>();
		train.addAll(l.subList(0, baseNum));
		for(int j=0;j<train.size();j++)
			l.remove(0);
		//add a max feature into the last sample
		LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
		train.add(tempL);

		//run experiment step by step
		for(int k=baseNum;l.size()>=stepSize;k+=stepSize)
		{
			double acc=0.0;
				
			//train the adaboost
			AdaBoost ab=new AdaBoost(numClassifiers, train);
			ab.train();
			//test the adaboost
			acc=ab.test(te);
			out.write(acc+"\t");
			out.flush();
			System.out.println("Trained on "+(train.size()-1)+" tweets ( "+(double)train.size()/(double)total+" of all the data ) with acc: "+acc);
				
			train.addAll(l.subList(0, stepSize));
			for(int j=0;j<stepSize;j++)
				l.remove(0);
			System.gc();
			}
		out.close();
		}
	}
