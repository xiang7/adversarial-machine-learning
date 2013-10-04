import java.io.*;
import java.util.*;

import edu.stanford.nlp.ling.Labeled;

import jnisvmlight.LabeledFeatureVector;
public class Tool {
	
	/**
	 * Read feature vectors into a vector of labled feature vectors
	 * ID \t label feaNum:1.0 feaNum:1.0 ...
	 * @param file
	 * @return
	 * @throws Exception
	 */
	public static List<LabeledFeatureVector> readFeaVec(File file) throws Exception
	{
		Vector<LabeledFeatureVector> v=new Vector<LabeledFeatureVector>();
		BufferedReader in= new BufferedReader(new FileReader(file));
		String temp="";
		while((temp=in.readLine())!=null)
		{
			String[] split=temp.split(ConstantClass.delimiter);
			if(split.length!=2)
				continue;
			String vec=split[1];
			String[] vecSplit=vec.split(" ");
			int[] fea = new int[vecSplit.length-1];
			double[] val = new double[vecSplit.length-1];
			int k = 0;
			for (int i=1;i<vecSplit.length;i++) {
				String[] tempNum=null;
				try {
					tempNum = vecSplit[i].split(":");
					fea[k] = Integer.valueOf(tempNum[0]);
					val[k++] = Double.valueOf(tempNum[1]);
				} catch (Exception e) {
					System.out.println(temp);	
					System.exit(0);
				}
			}
			LabeledFeatureVector l = new LabeledFeatureVector(
					Double.valueOf(vecSplit[0]), fea, val);
			v.add(l);
		}
			in.close();
			return v;
		
	}

	/**
	 * Get num of random vectors from v - the vectors are copied
	 * @param v
	 * @param num
	 * @return
	 * @throws Exception
	 */
	public static List<LabeledFeatureVector> getRandVec(List<LabeledFeatureVector> original, int num, boolean withReplacement) throws Exception
	{
		List<LabeledFeatureVector> v=new Vector<LabeledFeatureVector>();
		List<LabeledFeatureVector> l=new Vector<LabeledFeatureVector>();
		v.addAll(original);
		for(int i=0;i<num;i++)
		{
			int r=(int)Math.floor(Math.random()*v.size());
			LabeledFeatureVector temp=v.get(r);
			int[] dims=new int[temp.size()];
			double[] vals=new double[temp.size()];
			for(int j=0;j<temp.size();j++)
			{
				dims[j]=temp.getDimAt(j);
				vals[j]=temp.getValueAt(j);
			}
			LabeledFeatureVector newTemp=new LabeledFeatureVector(temp.getLabel(),dims,vals);
			l.add(newTemp);
			if(!withReplacement)
				v.remove(r);
		}
		return l;
	}
	
	//public static List<LabeledFeatureVector> getRandVec()
	
	/**
	 * Get num of random vectors from v - the vectors are not copied but passed back
	 * @param v
	 * @param num
	 * @return
	 * @throws Exception
	 */
	public static List<LabeledFeatureVector> getRandVecByReference(List<LabeledFeatureVector> v, int num) throws Exception
	{
		List<LabeledFeatureVector> l=new Vector<LabeledFeatureVector>();
		for(int i=0;i<num;i++)
		{
			int r=(int)Math.floor(Math.random()*v.size());
			l.add(v.get(r));
		}
		return l;
	}
	
	/**
	 * Get num of random vectors from v, with these removed from the original list
	 * @param v
	 * @param num
	 * @return
	 * @throws Exception
	 */
	public static List<LabeledFeatureVector> getRandVecRemove(List<LabeledFeatureVector> v, int num) throws Exception
	{
		List<LabeledFeatureVector> l=new Vector<LabeledFeatureVector>();
		for(int i=0;i<num;i++)
		{
			int r=(int)Math.floor(Math.random()*v.size());
			LabeledFeatureVector temp=v.get(r);
			int[] dims=new int[temp.size()];
			double[] vals=new double[temp.size()];
			for(int j=0;j<temp.size();j++)
			{
				dims[j]=temp.getDimAt(j);
				vals[j]=temp.getValueAt(j);
			}
			LabeledFeatureVector newTemp=new LabeledFeatureVector(temp.getLabel(),dims,vals);
			l.add(newTemp);
			v.remove(r);
		}
		return l;
	}
	
	public static List<LabeledFeatureVector> getWieghtedSampleWithoutRp(List<LabeledFeatureVector> all,List<Double> w_temp,int num) throws Exception
	{
		List<LabeledFeatureVector> allVec=new Vector<LabeledFeatureVector>(all);
		List<Double> w=new Vector<Double>(w_temp);
		List<LabeledFeatureVector> re=new Vector<LabeledFeatureVector>();
		if(num>allVec.size())
		{
			System.out.println("11");
			return null;
		}
		if(num==allVec.size())
			return allVec;
		if(w.size()!=allVec.size())
			{
			System.out.println("12");
			return null;
			}
		//normal case
		for(int i=0;i<num;i++)
		{
			//normalize the vector and select the one to remove
			double sum=sum(w,0,w.size());
			double r=Math.random();
			int j=0;
			int break_point=-1;
			double temp_sum=0.0;
			for(j=0;j<w.size();j++)
			{
				w.set(j,w.get(j)/sum);
				temp_sum=temp_sum+w.get(j);
				if(r-temp_sum<0&&break_point==-1)
					break_point=j;
			}
			
			//remove the one and copy into the result list
			LabeledFeatureVector temp=allVec.remove(break_point);
			w.remove(break_point);
			int[] dims=new int[temp.size()];
			double[] vals=new double[temp.size()];
			for(j=0;j<temp.size();j++)
			{
				dims[j]=temp.getDimAt(j);
				vals[j]=temp.getValueAt(j);
			}
			LabeledFeatureVector newTemp=new LabeledFeatureVector(temp.getLabel(),dims,vals);
			re.add(newTemp);
			//update the weight vector
			for(j=0;j<w.size();j++)
			{
				w.set(j, w.get(j)/(1-w.get(j)));
			}
		}
		return re;
	}
	
	public static double sum(List<Double> w,int from,int to)
	{
		double sum=0.0;
		for(int i=from;i<to;i++)
			sum+=w.get(i);
		return sum;
	}
	
	public static void main(String args[]) throws Exception
	{
		Hashtable<Integer, Integer> t=new Hashtable<Integer, Integer>();
		List<Double> w=new Vector<Double>();
		List<LabeledFeatureVector> l=new Vector<LabeledFeatureVector>();
		for(int i=0;i<10000;i++)
			{
			w.add(1.0);
			l.add(new LabeledFeatureVector(i, 1));
			}
		w.set(100, 100.0);
		w.set(5000, 100.0);
		w.set(9000, 100.0);
		for(int i=0;i<10;i++)
		{
			List<LabeledFeatureVector> tem=getWieghtedSampleWithoutRp(l, w, 2500);
			for(LabeledFeatureVector tempL : tem)
			{
				if(t.containsKey((int)tempL.getLabel()))
					t.put((int)tempL.getLabel(), t.get((int)tempL.getLabel())+1);
				else
					t.put((int)tempL.getLabel(), 1);
			}
		}
		for(int i=0;i<10000;i++)
			{
				if(t.containsKey(i))
					System.out.println(t.get(i));
				else
					System.out.println(0);
					
			}
	}
	
	public static double weight_change_ratio(double[] w1, double[] w2) throws Exception
	{
		if(w1==null)
			{
			w1=new double[w2.length];
			for(int i=0;i<w1.length;i++)
				w1[i]=0;
			}
		double w_neg=0.0;
		double w_pos=0.0;
		for(int i=0;i<w1.length&&i<w2.length;i++)
		{
			double w_temp=(w2[i]-w1[i])*w1[i];
			if(w_temp<0)
				w_neg+=w_temp*w_temp;
			else
				w_pos+=w_temp*w_temp;
		}
		return w_neg/(w_neg+w_pos);
	}
	
	public static double angular_diff(double[] w1,double[] w2)
	{
		if(w1==null)
			return 0.0;
		double prod=0.0;
		double w1_norm=0.0;
		double w2_norm=0.0;
		for(int i=0;i<w1.length;i++)
		{
			w1_norm+=w1[i]*w1[i];
			w2_norm+=w2[i]*w2[i];
			prod+=w1[i]*w2[i];
		}
		return Math.acos(prod/Math.sqrt(w2_norm)/Math.sqrt(w1_norm));
	}
}
