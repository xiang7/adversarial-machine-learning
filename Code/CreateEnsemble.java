import java.io.BufferedWriter;

import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Vector;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
import weka.classifiers.bayes.NaiveBayesMultinomial;


public class CreateEnsemble {
	
	private static int stepSize=200;
	private static double start_point=0.5;
	private static double pctGood=0.5;
	private static int end_point=(int)(ConstantClass.DataSetSize*(start_point+0.5)+stepSize);
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{

		run(	15,
				(int)(start_point*ConstantClass.DataSetSize+56*stepSize),
				new File("I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\CCS2013\\Data\\FINAL\\Egyptian\\EnsembleCreate\\CREATEVECSVM50"),
				new File(ConstantClass.TestVec),
				new File("I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\CCS2013\\Data\\FINAL\\Egyptian\\EnsembleCreate\\CREATEACC"),
				new File("I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\CCS2013\\Data\\FINAL\\Egyptian\\EnsembleCreate\\CREATEAG"),
				new File("I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\CCS2013\\Data\\FINAL\\Egyptian\\EnsembleCreate\\CREATETIME"),
				1,
				false);
		
	}
	
	public static void run(int numClassifiers,int baseNum,File trainFile,File testFile,File outFile,File outAve,File outTimeFile,int num,boolean withReplacement) throws Exception
	{
		//read in all the training vectors
		BufferedWriter out=new BufferedWriter(new FileWriter(outFile,true));
		BufferedWriter outAgree=new BufferedWriter(new FileWriter(outAve,true));
		BufferedWriter outTime=new BufferedWriter(new FileWriter(outTimeFile,true));
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
		for(int i=0;i<num;i++)
		{
			System.out.println(i+"th run");
			//generate baseNum of tweets
			List<LabeledFeatureVector> baseNumTwt=new Vector<LabeledFeatureVector>();
			baseNumTwt.addAll(l.subList(0, baseNum));
			for(int j=0;j<baseNumTwt.size();j++)
				l.remove(0);
			//add a max feature into the last sample
			LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			//train.add(tempL);

			
			//run experiment step by step
			for(int k=baseNum;l.size()>=stepSize;k+=stepSize)
			{
				long start_time=System.currentTimeMillis();
				//get the train tweets
				List<List<LabeledFeatureVector>> train=new Vector<List<LabeledFeatureVector>>();
				for(int j=0;j<numClassifiers;j++)
				{
					List<LabeledFeatureVector> tempTrain=new Vector<LabeledFeatureVector>();
					tempTrain.addAll(Tool.getRandVec(baseNumTwt, (int)(k*pctGood),withReplacement));
					tempTrain.add(tempL);
					train.add(tempTrain);
				}
				
				SVMLightModel[] svms=new SVMLightModel[numClassifiers];
				double acc=0.0;
				
				//train the ensemble
				for(int j=0;j<numClassifiers;j++)
				{
					LabeledFeatureVector[] tr=train.get(j).toArray(new LabeledFeatureVector[train.get(j).size()]);
					SVMLightModel m=null;
					NaiveBayesMultinomial nb=null;
					if(ConstantClass.currClassifier==ConstantClass.SVM)
					{
						m=TrainTestOut.train(tr);
						svms[j]=m;
						}
					else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
					{
						nb=TrainTestOut.trainBayes(tr);
						//	acc=TrainTestOut.testBayes(nb,te);
							}
				}
				long end_time=System.currentTimeMillis();
				//test the ensemble
				double[] result=TrainTestOut.testSVMEnsemble(svms, te);
				acc=result[0];
				double agree=result[1];
				out.write(acc+"\t");
				outAgree.write(agree+"\t");
				outTime.write((end_time-start_time)+"\t");
				out.flush();
				outTime.flush();
				outAgree.flush();
				System.out.println("Trained on "+(train.get(0).size()-1)+" tweets ( "+(double)train.get(0).size()/(double)total+" of all the data ) with acc: "+acc+"\tBase tweet size: "+baseNumTwt.get(0).size()+" "+baseNumTwt.get(1).size());
				
				//bad tweets
				//List<List<LabeledFeatureVector>> badTwts=new Vector<List<LabeledFeatureVector>>();
				baseNumTwt.addAll(l.subList(0, stepSize));
				for(int j=0;j<stepSize;j++)
					l.remove(0);
				System.gc();
				
			}
					out.write("\r\n");
					outAgree.write("\r\n");
					outTime.write("\r\n");
					out.flush();
					outAgree.flush();
					outTime.flush();
			}

				out.close();
				outTime.close();
				outAgree.close();
	}

}
