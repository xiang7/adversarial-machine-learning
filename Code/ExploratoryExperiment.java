import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Vector;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;


public class ExploratoryExperiment {
	//Election: 1200, {6,30,60,300,600,1200}
	//Egyptian: 3000, {6,30,60,300,600,1200}
	private static int stepSize=2000;
	private static double start_point=0.25;
	private static int[] startingPoints=new int[] {6,30,60,300,600,1200,1800};

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		for(int i=1;i<2;i++) {
			if(i==1)
				start_point=0.5;
			else if(i==2)
				start_point=0.75;
		/**
		 * seen
		 *//*
		run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0.5,0.5},
				true,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Exploratory//Seen"),
				5,
				(int)(ConstantClass.DataSetSize));*/
		/**
		 * unseen
		 */
		run((int)(start_point*ConstantClass.DataSetSize),
				new File[] {new File(ConstantClass.folder+"posVec"),new File(ConstantClass.folder+"negVec")},
				new double[] {0.5,0.5},
				new double[] {0.5,0.5},
				true,
				new File(ConstantClass.TestVec),
				new File(ConstantClass.folder+String.valueOf((int)(start_point*100))+"Exploratory//Seen"),
				5,
				(int)(ConstantClass.DataSetSize));//ConstantClass.DataSetSize
	}
	}


	
	
	/**
	 * Run the exploratory experiment
	 * @param baseNum - the number of tweets to train the victim classifier
	 * @param trainFile
	 * @param basePctg - the percentage of tweets from each class to train the victim classifier
	 * @param repPctg - the percentage of tweets from each class to train the new classifiers
	 * @param seen - whether to use the tweets seen by the victim classifier or not
	 * @param testFile
	 * @param outFile - the file to write the results
	 * @param num - number of iterations
	 * @param maxNum - maximum number of tweets to train the new classifiers
	 * @throws Exception
	 */
	public static void run(int baseNum,File[] trainFile,double[] basePctg,double[] repPctg,boolean seen,File testFile,File outFile,int num, int maxNum) throws Exception
	{
		//read in all the training vectors
		BufferedWriter out=new BufferedWriter(new FileWriter(outFile));
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
			System.out.println(i+"th run");
			//generate baseNum of tweets
			List<List<LabeledFeatureVector>> baseNumTwt=new Vector<List<LabeledFeatureVector>>();
			List<LabeledFeatureVector> train=new Vector<LabeledFeatureVector>();
			//add a max feature into the last sample
			LabeledFeatureVector tempL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			train.add(tempL);
			//copy l for unseen case
			List<List<LabeledFeatureVector>> lCopy=new Vector<List<LabeledFeatureVector>>();
			if(!seen)
				for(int j=0;j<trainFile.length;j++)
					lCopy.add(new Vector<LabeledFeatureVector>(l.get(j)));
			//get the base training data
			for(int j=0;j<trainFile.length;j++)
			{
				if(seen)
					baseNumTwt.add(Tool.getRandVec(l.get(j), (int)(baseNum*basePctg[j]),false));
				else
					baseNumTwt.add(Tool.getRandVecRemove(lCopy.get(j), (int)(baseNum*basePctg[j])));
				train.addAll(baseNumTwt.get(j));
			}
			//train the base svm
			SVMLightModel baseSVM=null;
			MyNaiveBayesMultinomial baseNb=null;
			if(ConstantClass.currClassifier==ConstantClass.SVM)
				baseSVM= TrainTestOut.train(train.toArray(new LabeledFeatureVector[train.size()]));
			else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
				baseNb= TrainTestOut.trainBayes(train.toArray(new LabeledFeatureVector[train.size()]));
			
			//training vectors for the new svms
			List<LabeledFeatureVector> newTrain=new Vector<LabeledFeatureVector>();
			//add max feature into the training vectors of new svms
			LabeledFeatureVector tempLNew=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			newTrain.add(tempLNew);
			
			for(int m=1;m<points.size();m++)
			{
				int k=points.elementAt(m);
				//prepare noisy training data
				if(seen) {
					for(int j=0;j<trainFile.length;j++)
						if(ConstantClass.currClassifier==ConstantClass.SVM)
							newTrain.addAll(TrainTestOut.classify(baseSVM, Tool.getRandVec(baseNumTwt.get(j), (int)((k-points.elementAt(m-1))*repPctg[j]),true)));
						else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
						{
							newTrain.addAll(TrainTestOut.classifyBayes(baseNb, Tool.getRandVec(baseNumTwt.get(j), (int)((k-points.elementAt(m-1))*repPctg[j]),true)));
							System.out.println(TrainTestOut.testBayes(baseNb, Tool.getRandVec(baseNumTwt.get(j), (int)((k-points.elementAt(m-1))*repPctg[j]),true)));
						}
					}
					else {
						for(int j=0;j<trainFile.length;j++)
							if(ConstantClass.currClassifier==ConstantClass.SVM)
								newTrain.addAll(TrainTestOut.classify(baseSVM, Tool.getRandVec(lCopy.get(j), (int)((k-points.elementAt(m-1))*repPctg[j]),true)));
							else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
							{
								newTrain.addAll(TrainTestOut.classifyBayes(baseNb, Tool.getRandVec(lCopy.get(j), (int)((k-points.elementAt(m-1))*repPctg[j]),true)));
								System.out.println(TrainTestOut.testBayes(baseNb, Tool.getRandVec(lCopy.get(j), (int)((k-points.elementAt(m-1))*repPctg[j]),true)));
							}
					}
				
				SVMLightModel newSVM=null;
				MyNaiveBayesMultinomial newNb=null;
				double acc=0.0;
				if(ConstantClass.currClassifier==ConstantClass.SVM)
					{
					newSVM=TrainTestOut.train(newTrain.toArray(new LabeledFeatureVector[newTrain.size()]));
					acc=TrainTestOut.test(newSVM, te);
					}
				else if(ConstantClass.currClassifier==ConstantClass.NAIVEBAYES)
					{
					newNb=TrainTestOut.trainBayes(newTrain.toArray(new LabeledFeatureVector[newTrain.size()]));
					acc=TrainTestOut.testBayes(newNb, te);
							}
				out.write(acc+"\t");
				out.flush();
				System.out.println("Trained on "+(newTrain.size()-1)+" tweets ( "+(double)newTrain.size()/(double)total+" of all the data ) with acc: "+acc);
			}
			out.write("\r\n");
			out.flush();
	}

		out.close();
}

}
