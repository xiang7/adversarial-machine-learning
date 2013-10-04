import jnisvmlight.*;

import java.util.List;
import weka.core.*;
import weka.classifiers.bayes.*;
import java.io.*;

public class TrainTestOut {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
	}
	
	
	public static MyNaiveBayesMultinomial trainBayes(LabeledFeatureVector[] feaVec) throws Exception
	{
		MyNaiveBayesMultinomial nb=new MyNaiveBayesMultinomial();
		nb.buildClassifier(vecToInstances(feaVec));
		return nb;
	}
	public static List<LabeledFeatureVector> classifyBayes(NaiveBayesMultinomial nb,List<LabeledFeatureVector> feaVec) throws Exception
	{
		Instances is=vecToInstances(feaVec);
		for(int i=0;i<is.numInstances();i++)
		{
			double re=nb.classifyInstance(is.instance(i));
			if((int)re==0)
				feaVec.get(i).setLabel(1);
			else if((int)re==1)
				feaVec.get(i).setLabel(-1);
			else
				System.out.println("Unrecognized label");
		}
		return feaVec;
	}
	public static double testBayes(MyNaiveBayesMultinomial nb,List<LabeledFeatureVector> feaVec) throws Exception
	{
		return testBayes(nb, feaVec.toArray(new LabeledFeatureVector[feaVec.size()]));
	}
	public static double testBayes(MyNaiveBayesMultinomial nb,LabeledFeatureVector[] feaVec) throws Exception
	{
		int correct=0;
		Instances is=vecToInstances(feaVec);
		for(int i=0;i<is.numInstances();i++)
		{
			double cl=nb.classifyInstance(is.instance(i));
			if(feaVec[i].getLabel()>0&&(int)cl==0||feaVec[i].getLabel()<0&&(int)cl==1)
				correct++;
		}
		return (double)correct/(double)feaVec.length;
	}
	public static Instances vecToInstances(List<LabeledFeatureVector> feaVec) throws Exception
	{
		return vecToInstances(feaVec.toArray(new LabeledFeatureVector[feaVec.size()]));
	}
	public static Instances vecToInstances(LabeledFeatureVector[] feaVec) throws Exception
	{
		FastVector fv=new FastVector(ConstantClass.MaxNumFea+2);
		for(int i=0;i<=ConstantClass.MaxNumFea;i++)
		{
			fv.addElement(new Attribute(String.valueOf(i)));
		}
		FastVector classVec=new FastVector(2);
		classVec.addElement("pos");
		classVec.addElement("neg");
		fv.addElement(new Attribute("class",classVec));
		Instances instces=new Instances("NB",fv,feaVec.length);
		instces.setClassIndex(fv.size()-1);
		
		for(int i=0;i<feaVec.length;i++)
		{
			double attValues[]=new double[feaVec[i].size()];
			int indices[]=new int[feaVec[i].size()];
			for(int j=0;j<indices.length;j++)
			{
				attValues[j]=feaVec[i].getValueAt(j);
				indices[j]=feaVec[i].getDimAt(j);
			}
			SparseInstance si=new SparseInstance(1.0, attValues, indices, attValues.length);
			si.setDataset(instces);
			if(feaVec[i].getLabel()>0)
				si.setClassValue("pos");
			else
				si.setClassValue("neg");
			instces.add(si);
		}
		return instces;
	}
	
	/**
	 * Train on feaVec.
	 * @param feaVec
	 * @return
	 * @throws Exception
	 */
	public static SVMLightModel train(LabeledFeatureVector[] feaVec) throws Exception
	{
		SVMLightInterface svm=new SVMLightInterface();
		SVMLightInterface.SORT_INPUT_VECTORS=false;
		TrainingParameters tp=new TrainingParameters();
		SVMLightModel model=svm.trainModel(feaVec,tp);
		return model;
	}
	
	/**
	 * Test feaVec using model. The feaVec comes with labels
	 * @param model
	 * @param feaVec
	 * @return
	 * @throws Exception
	 */
	public static double test(SVMLightModel model,LabeledFeatureVector[] feaVec) throws Exception
	{
		int correct=0;
		for(LabeledFeatureVector l : feaVec)
		{
			double re=model.classify(l);
		if((int)(l.getLabel())==1&&re>=0||(int)(l.getLabel())==-1&&re<0) correct++;
		}
		return (double)correct/(double)feaVec.length;
	}
	
	public static double[] testSVMEnsemble(SVMLightModel[] svms,LabeledFeatureVector[] feaVec) throws Exception
	{
		int correct=0;
		int totalAgree=0;
		for(LabeledFeatureVector l:feaVec)
		{
			int pos=0;
			int neg=0;
			for(int j=0;j<svms.length;j++)
				if(svms[j].classify(l)>0)
					pos++;
				else
					neg++;
			if(pos>neg&&l.getLabel()>0)
				{
				correct++;
				totalAgree+=pos;
				}
			else if(pos<neg&&l.getLabel()<0)
				{
				totalAgree+=neg;
				correct++;
				}
		}
		
		return new double[] {(double)correct/(double)feaVec.length,(double)totalAgree/(double)correct};
	}
	
	/**
	 * Classify feaVec using model. The labels are in feaVec[i].label
	 * @param model
	 * @param feaVec
	 * @return
	 * @throws Exception
	 */
	public static LabeledFeatureVector[] classify(SVMLightModel model,LabeledFeatureVector[] feaVec) throws Exception
	{
		for(LabeledFeatureVector l : feaVec)
			l.setLabel(model.classify(l));
		return feaVec;
	}
	
	public static List<LabeledFeatureVector> classify(SVMLightModel model,List<LabeledFeatureVector> feaVec) throws Exception
	{
		for(LabeledFeatureVector l : feaVec)
			l.setLabel(model.classify(l));
		return feaVec;
	}
}
