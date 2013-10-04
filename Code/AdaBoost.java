import java.io.BufferedWriter;

import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Vector;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;
/**
 * The class to do AdaBoost
 * @author 6
 *
 */
public class AdaBoost {
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}

	private double sampleSize=0.5;
	private SVMLightModel[] ensemble=null;
	private List<Double> w=null;
	private int k_max=3;
	private List<LabeledFeatureVector> train=null;
	private double alpha[]=null;
	
	public AdaBoost(int k_max,List<LabeledFeatureVector> train)
	{
		this.k_max=k_max;
		this.train=train;
		this.w=new Vector<Double>();
		this.alpha=new double[k_max];
		ensemble=new SVMLightModel[k_max];
		//initialize w to be 1/n for each example
		for(int i=0;i<train.size();i++)
			w.add((double)1/(double)train.size());
	}
	
	
	public void train() throws Exception
	{
		for(int i=0;i<k_max;i++)
		{
			//select from the training set a distribution
			List<LabeledFeatureVector> tempL=Tool.getWieghtedSampleWithoutRp(train, w, (int)(train.size()*sampleSize));
			LabeledFeatureVector addL=new LabeledFeatureVector(1.0,new int[] {ConstantClass.MaxNumFea},new double[] {1.0});
			tempL.add(addL);
			//train the classifier
			SVMLightInterface svm=new SVMLightInterface();
			SVMLightInterface.SORT_INPUT_VECTORS=false;
			TrainingParameters tp=new TrainingParameters();
			ensemble[i]=svm.trainModel(tempL.toArray(new LabeledFeatureVector[tempL.size()]),tp);
			//test the training set, get the error rate
			int wrong=0;
			for(LabeledFeatureVector temp:train)
				if(ensemble[i].classify(temp)*temp.getLabel()<0)
					wrong++;
			double err=(double)wrong/(double)train.size();
			alpha[i]=0.5*Math.log((1-err)/err);
			//update weights
			double cor_update=Math.pow(Math.E, -alpha[i]);
			double err_update=Math.pow(Math.E, alpha[i]);
			for(int j=0;j<w.size();j++)
				if(ensemble[i].classify(train.get(j))*train.get(j).getLabel()>0)
					w.set(j, cor_update*w.get(j));
				else
					w.set(j, err_update*w.get(j));
		}
		//for(double k : alpha)
		//	System.out.print(k+" ");
		//System.out.print("\n");
		BufferedWriter out=new BufferedWriter(new FileWriter(ConstantClass.folder+"w"+train.size()));
		for(double k : w)
			out.write(k+"\t");
		out.flush();
		out.close();
	}
	
	public double test(LabeledFeatureVector[] l)
	{
		double acc=0.0;
		for(LabeledFeatureVector tempL:l)
		{
			double out=0.0;
			for(int i=0;i<k_max;i++)
				if(ensemble[i].classify(tempL)>0)
					out+=alpha[i];
				else
					out+=-alpha[i];
			if(out*tempL.getLabel()>0)
				acc+=1.0;
		}
		return acc/(double)l.length;
	}
	
	public LabeledFeatureVector[] classify(LabeledFeatureVector[] l)
	{
		for(LabeledFeatureVector tempL:l)
		{
			double out=0.0;
			for(int i=0;i<k_max;i++)
				if(ensemble[i].classify(tempL)>0)
					out+=alpha[i];
				else
					out+=-alpha[i];
			if(out*tempL.getLabel()>0)
				tempL.setLabel(1.0);
			else
				tempL.setLabel(-1.0);
		}
		return l;
	}

}