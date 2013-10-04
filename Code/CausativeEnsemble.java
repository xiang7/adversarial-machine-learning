import java.io.BufferedWriter;

import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Vector;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
import weka.classifiers.bayes.NaiveBayesMultinomial;


public class CausativeEnsemble {
	
	private static int stepSize=2000;
	private static double start_point=0.5;
	private static double pctGood=0.5;
	private static int end_point=(int)(ConstantClass.DataSetSize*(start_point+0.5)+stepSize);
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{

		run(	15,
				(int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0.5,0.5},
				true,
				false,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//EnsembleReplicateFlip"),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//EnsembleReplicateFlipAgree"),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//EnsembleReplicateFlipTime"),
				5,
				(int)(end_point),
				false);
		
		/*run(	1,
				(int)(start_point*ConstantClass.DataSetSize),
				(int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0.5,0.5},
				true,
				true,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//TestReplicateFlip"),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//TestReplicateFlipAgree"),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Causative//TestReplicateFlipTime"),
				5,
				(int)(end_point));*/
	}
	
	public static void run(int numClassifiers,int baseNum,File[] trainFile,double[] basePctg,double[] repPctg,boolean flip,boolean seen,File testFile,File outFile,File outAve,File outTimeFile,int num, int maxNum,boolean withReplacement) throws Exception
	{
		//read in all the training vectors
		BufferedWriter out=new BufferedWriter(new FileWriter(outFile));
		BufferedWriter outAgree=new BufferedWriter(new FileWriter(outAve));
		BufferedWriter outTime=new BufferedWriter(new FileWriter(outTimeFile));
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
		//test tweets
		LabeledFeatureVector[] te=t.toArray(new LabeledFeatureVector[t.size()]);
				
		//train and test
		for(int k=baseNum;k<maxNum;k+=stepSize)
			out.write(k+"\t");
		out.write("\n");
		for(int k=baseNum;k<maxNum;k+=stepSize)
			outAgree.write(k+"\t");
		outAgree.write("\n");
		for(int k=baseNum;k<maxNum;k+=stepSize)
			outTime.write(k+"\t");
		outTime.write("\n");
		
		for(int i=0;i<num;i++)
		{
			System.out.println(i+"th run");
			//generate baseNum of tweets
			List<List<LabeledFeatureVector>> baseNumTwt=new Vector<List<LabeledFeatureVector>>();
			
			List<List<LabeledFeatureVector>> lCopy=new Vector<List<LabeledFeatureVector>>();
			if(!seen)
				for(int j=0;j<trainFile.length;j++)
					lCopy.add(new Vector<LabeledFeatureVector>(l.get(j)));
			//add a max feature into the last sample
			LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			//train.add(tempL);
			//get the base training data
			for(int j=0;j<trainFile.length;j++)
			{
				if(seen)
					baseNumTwt.add(Tool.getRandVec(l.get(j), (int)(baseNum*basePctg[j]),false));
				else
					baseNumTwt.add(Tool.getRandVecRemove(lCopy.get(j), (int)(baseNum*basePctg[j])));
			}
			
			//run experiment step by step
			for(int k=baseNum;k<maxNum;k+=stepSize)
			{
				long start_time=System.currentTimeMillis();
				//get the train tweets
				List<List<LabeledFeatureVector>> train=new Vector<List<LabeledFeatureVector>>();
				for(int j=0;j<numClassifiers;j++)
				{
					List<LabeledFeatureVector> tempTrain=new Vector<LabeledFeatureVector>();
					for(int jj=0;jj<trainFile.length;jj++)
						tempTrain.addAll(Tool.getRandVec(baseNumTwt.get(jj), (int)(k*pctGood*basePctg[jj]),withReplacement));
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
						//acc=TrainTestOut.test(m,te);
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
				for(int j=0;j<trainFile.length;j++)
				{
					if(seen)
						if(flip)
							baseNumTwt.get(j).addAll(CausativeExperiment.flip(Tool.getRandVec(baseNumTwt.get(j).subList(0, (int)(baseNum*basePctg[j])), (int)(stepSize*repPctg[j]),true)));
						else
							baseNumTwt.get(j).addAll(Tool.getRandVec(baseNumTwt.get(j).subList(0, (int)(baseNum*basePctg[j])), (int)(stepSize*repPctg[j]),true));
					else
						if(flip)
							baseNumTwt.get(j).addAll(CausativeExperiment.flip(Tool.getRandVec(lCopy.get(j), (int)(stepSize*repPctg[j]),true)));
						else
							baseNumTwt.get(j).addAll(Tool.getRandVec(lCopy.get(j), (int)(stepSize*repPctg[j]),true));
				}
				
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
