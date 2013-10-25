import java.util.*;
import java.io.*;

import jnisvmlight.LabeledFeatureVector;

import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

/**
 * This class extracts features from preprocessed tweets
 * 
 * @author Luojie Xiang
 * 
 */
public class FeatureExtractor {

	private static Hashtable<String, Integer> table=new Hashtable<String, Integer>();
	private static int feaNum=1;
	private static boolean POS = false; //part of speech tagging
	private static MaxentTagger mt = null;

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		
		extractFeatures(
				new File(ConstantClass.FeatureFile),
				new File[] {new File(ConstantClass.folder+"pos2"),
				new File(ConstantClass.folder+"neg2"),
				new File(ConstantClass.folder+"posTest"),
				new File(ConstantClass.folder+"negTest"),
				new File(ConstantClass.folder+"posAttack"),
				new File(ConstantClass.folder+"negAttack")
				},
				new File[] {new File(ConstantClass.folder+"pos2Vec"),
			new File(ConstantClass.folder+"neg2Vec"),
			new File(ConstantClass.folder+"posTestVec"),
			new File(ConstantClass.folder+"negTestVec"),
				new File(ConstantClass.folder+"posAttackVec"),
				new File(ConstantClass.folder+"negAttackVec")
				},
			new String[] {"1","-1","1","-1","1","-1"},POS);
		/*
		extractFeatures(
				new File(ConstantClass.FeatureFile),
				new File[] {new File(ConstantClass.folder+"labeledPosTest200"),
				new File(ConstantClass.folder+"labeledNegTest200")},
				new File[] {new File(ConstantClass.folder+"posTestVec"),
			new File(ConstantClass.folder+"negTestVec")},
			new String[] {"1","-1"},POS);
			*/
	}
	
	/**
	 * Set the POS field. If POS set, then initialize a POS tagger
	 * @param POS
	 * @throws Exception
	 */
	public static void setPOS(boolean POS) throws Exception
	{
		if (POS)
			{
			mt = new MaxentTagger(ConstantClass.tagger);
			FeatureExtractor.POS=true;
			}
		else
			FeatureExtractor.POS=false;
	}

	/**
	 * The method extract the feature from the string s and return a labeled
	 * feature vector with label specified. The method supports POS.
	 * 
	 * @param s
	 *            - the tweet
	 * @param label
	 *            - the label of the tweet
	 * @param pos
	 *            - use pos or not
	 * @return
	 */
	private static LabeledFeatureVector unigram(String s, String label,
			boolean pos) {
		TreeSet<Integer> ts = new TreeSet<Integer>();
		String[] split = s.split(" |\t");
		ArrayList<TaggedWord> tagged = new ArrayList<TaggedWord>();
		for (String temp : split) {
			if (temp.length() != 0)
				tagged.add(new TaggedWord(temp.trim()));
		}
		if (pos)
			tagged = mt.apply(tagged);
		for (TaggedWord temp : tagged) {
			if (pos) {
				if (!table.containsKey(temp.toString("_"))) 
					table.put(temp.toString("_"), feaNum++);
				ts.add(table.get(temp.toString("_")));
			} else {
				if (!table.containsKey(temp.word())) 
					table.put(temp.word(), feaNum++);
				ts.add(table.get(temp.word()));
			}
		}
		int[] fea = new int[ts.size()];
		double[] val = new double[ts.size()];
		int k = 0;
		for (int tempNum : ts) {
			fea[k] = tempNum;
			val[k++] = 1.0;
		}
		LabeledFeatureVector l = new LabeledFeatureVector(
				Double.valueOf(label), fea, val);
		return l;
	}

	/**
	 * Extract features from all files in tweetFile and write the 
	 * labeled feature vectors into outFiles with their corresponding
	 * labels specified by labels.
	 * The output also includes the ID of each tweet
	 * <br/>
	 * ID + \t + Label + + feature number:feature value + + ...
	 * @param feaFile
	 * @param tweetFile
	 * @param outFile
	 * @param labels
	 * @param pos
	 * @throws Exception
	 */
	public static void extractFeatures(File feaFile,File[] tweetFile,File[] outFile,String[] labels,boolean pos) throws Exception
	{
		//read in features into table
		feaNum=readFeaFile(feaFile);
		//for each tweet file, initiate a vector to maintain the labeled feature vectors
		Vector<Vector<String>> feaVec=new Vector<Vector<String>>();
		for(int i=0;i<tweetFile.length;i++)
			feaVec.add(new Vector<String>());
		//call unigram for each tweet and put the labeled fature vector into the correct vector
		for(int i=0;i<tweetFile.length;i++)
		{
			String temp="";
			BufferedReader in=new BufferedReader(new FileReader(tweetFile[i]));
			while((temp=in.readLine())!=null)
			{
				//take off the tweet ID
				String split[]=temp.split(ConstantClass.delimiter);
				feaVec.elementAt(i).add(split[0]+ConstantClass.delimiter+unigram(split[1],labels[i],pos).toString());
			}
			in.close();
		}
		//write the result out
		for(int i=0;i<tweetFile.length;i++)
		{
			BufferedWriter out=new BufferedWriter(new FileWriter(outFile[i]));
			for(String l : feaVec.elementAt(i))
				out.write(l+"");
			out.flush();
			out.close();
		}
		//write the feature out
		writeFeaFile(feaFile);
	}
	
	/**
	 * This method reads the features into the hash table
	 * @param feaFile
	 * @return - the count of features
	 * @throws Exception
	 */
	private static int readFeaFile(File feaFile) throws Exception
	{
		BufferedReader feaIn= new BufferedReader(new FileReader(feaFile));
		String temp="";
		int count=0;
		while((temp=feaIn.readLine())!=null)
		{
			String[] split=temp.split(ConstantClass.delimiter);
			if(split.length!=2)
				continue;
			if(table.containsKey(split[0]))
				continue;
			table.put(split[0],	Integer.valueOf(split[1]));
			count++;
		}
		feaIn.close();
		return count>feaNum?count:feaNum;
	}

	/**
	 * This method writes the features from the has table out to the feaFile
	 * @param feaFile
	 * @return
	 * @throws Exception
	 */
	private static void writeFeaFile(File feaFile) throws Exception{
		//not in order
		BufferedWriter out=new BufferedWriter(new FileWriter(feaFile));
		Iterator<Map.Entry<String, Integer>> it=table.entrySet().iterator();
		System.out.println("Total number of features: "+table.size());
		while(it.hasNext())
			{
			Map.Entry<String, Integer> en=it.next();
			out.write(en.getKey()+ConstantClass.delimiter+en.getValue()+"\r\n");
			}
		out.flush();
		out.close();
	}
}
