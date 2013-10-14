import java.util.*;
import java.io.*;

/**
 * This class takes in tweets as input. It performs the following task: <br\>
 * Select relevant tweets <br\>
 * Preprocess tweets (including assigning ID to each tweet) <br\>
 * Classify tweets according to their noisy label <br\>
 * <br\>
 * Output format: ID \t Tweet
 * @author Luojie Xiang
 * 
 */
public class TweetProcessor {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

		List<String> posEmo = getEmo(new File(ConstantClass.PositiveEmoticon));
		List<String> negEmo = getEmo(new File(ConstantClass.NegativeEmoticon));
		List<String> allEmo = getEmo(new File(ConstantClass.PositiveEmoticon));
		allEmo.addAll(negEmo);
		List<Vector<String>> v = preprocessTweets(
				select(new File[] { new File(ConstantClass.folder+"tweets") }, allEmo),
				posEmo, negEmo);
		writeTweets(new File(ConstantClass.folder + "pos"), new File(
				ConstantClass.folder + "neg"),
				v.get(0).size()>v.get(1).size()?v.get(0).subList(0, v.get(1).size()):v.get(0),
				v.get(1).size()>v.get(0).size()?v.get(1).subList(0,v.get(0).size()):v.get(1));
//test shortenUsername
//		System.out.println(shortenUsername("@7599b55dbdce4144422b0056d5331a69eecab6cafeb38e73e23215ea5c407bc5 oke, kirain ilang taunya lupa"));
	}

	/**
	 * Select tweets relevant (containing the query terms specified)
	 * 
	 * @param file
	 *            - the file of all tweets
	 * @param queryTerms
	 *            - the tweets containing these queries will be returned
	 * @return the vector of relevant tweets
	 */
	public static Vector<String> select(File[] allTweetFile,
			List<String> queryTerms) throws Exception {
		Vector<String> rlvTwt = new Vector<String>();
		for (File f : allTweetFile) {
			BufferedReader inTwt = new BufferedReader(new FileReader(f));
			String temp = "";
			while ((temp = inTwt.readLine()) != null) {
				for (String s : queryTerms)
					if (temp.indexOf(s) != -1) {
						rlvTwt.add(temp);
						break;
					}
			}
			inTwt.close();
		}
		return rlvTwt;
	}

	/**
	 * Extracts text of a tweet (get rid of dates IDs).
	 * @param temp - the tweet from which the text is extracted
	 * @return - the extracted text from the tweet
	 * @throws Exception
	 */
	public static String extractText(String temp) throws Exception {
		if (ConstantClass.dataSetName == ConstantClass.EgyptianDataSet) {
			String split[] = temp.split("\t");
			//temp = split[4].toUpperCase(); //for preprocessing experiment, don't do the to upper case
			temp = split[4];
		} else if (ConstantClass.dataSetName == ConstantClass.ElectionDataSet)
			temp = temp.replaceAll(",", " ");
		//return temp.trim().toUpperCase(); //for preprocessing experiment, don't do the to upper case
		return temp.trim();
	}

	/**
	 * Process each tweet according to the Stanford Paper. Classify the tweets into pos and neg by emoticons
	 * @param v - the tweets
	 * @param posEmo - the positive emoticons
	 * @param negEmo - the negative emoticons
	 * @return - a vector containing two vectors of tweets
	 * @throws Exception
	 */
	public static List<Vector<String>> preprocessTweets(List<String> v,
			List<String> posEmo, List<String> negEmo) throws Exception {
		int count = 0;
		Hashtable<String, Integer> hash = new Hashtable<String, Integer>();
		List<Vector<String>> result = new Vector<Vector<String>>();
		Vector<String> posTwt = new Vector<String>();
		Vector<String> negTwt = new Vector<String>();

		for (String temp : v) {
			temp = extractText(temp);

			// check retweet
			if (temp.toUpperCase().startsWith("RT "))
				continue;
			temp = temp.replaceAll("\\s+", " ");
			if (temp.toUpperCase().indexOf(" RT ") != -1)
				continue;

			// check :P
			if (temp.toUpperCase().indexOf(":P") != -1)
				continue;

			// check emoticon and find out whether it is positive or negative
			// remove the emoticons
			boolean isPos = false;
			boolean isNeg = false;

			for (String emo : posEmo) {
				int idx = temp.indexOf(emo);
				if (idx != -1) {
					isPos = true;
					temp = temp.replaceAll(emo.replaceAll("\\)", "\\\\)"), " ");
				}
			}
			for (String emo : negEmo) {
				int idx = temp.indexOf(emo);
				if (idx != -1) {
					isNeg = true;
					temp = temp.replaceAll(emo.replaceAll("\\(", "\\\\("), " ");
				}
			}
			if (isPos && isNeg || !isPos && !isNeg)
				continue;

			temp = temp + " ";
			// change @ into USERNAME
			//temp = temp.replaceAll("@.*? ", "USERNAME ");
			if(temp.indexOf("@")!=-1)
				temp=shortenUsername(temp);

			// change urls into URL
			String urlValidChar="A-Za-z0-9\\-\\._~:\\/\\?\\#\\[\\]@!$&'\\(\\)\\*\\+,;=%";
			temp = temp.replaceAll("(HTTP|http|https|Http|Https|HTTPS)://["+urlValidChar+"]*", "URL ");

			// change multiple occurrences into 2
			//temp = temp.trim().replaceAll("([A-Z])\\1{2,}", "$1$1"); //not doing this for preprocessing experiment

			if (temp.split("\\s+").length <= 1)
				continue;

			temp = temp.replaceAll("\\s+", " ");
			if (temp.indexOf(" RT ") != -1)
				continue;

			//check repeated tweets
			if (hash.containsKey(temp))
				continue;
			else {
				count++;
				hash.put(temp, count);
			}

			// put the tweet into the hashtable with ID in front
			if (isPos)
				posTwt.add(hash.get(temp) + ConstantClass.delimiter + temp);
			else
				negTwt.add(hash.get(temp) + ConstantClass.delimiter + temp);
		}
		result.add(posTwt);
		result.add(negTwt);
		return result;
	}
	
	public static String shortenUsername(String s){
		final int normLen=64;
		int idx=s.indexOf("@");
		while(idx!=-1){
			if(s.length()<=idx+normLen)
				break;
			String temp=s.substring(idx+1,idx+normLen+1); //all anonymized user names are of length 66
			if(temp.matches("[0-9A-Za-z]{"+normLen+"}")){
				s=s.replaceAll(temp,"USER_"+temp.substring(0,6));
				idx=idx+6;
			}else
				idx++;
			idx=s.indexOf("@",idx);
		}	
		return s;
	}

	/**
	 * Write the result out to file
	 * @param posFile
	 * @param negFile
	 * @param posTwt
	 * @param negTwt
	 * @throws Exception
	 */
	public static void writeTweets(File posFile, File negFile,
			List<String> posTwt, List<String> negTwt) throws Exception { 
		BufferedWriter outPos = new BufferedWriter(new FileWriter(posFile));
		BufferedWriter outNeg = new BufferedWriter(new FileWriter(negFile));
		for (String s : posTwt)
			outPos.write(s + "\r\n");
		for (String s : negTwt)
			outNeg.write(s + "\r\n");
		System.out.println("Positive Tweet Written: " + posTwt.size());
		System.out.println("Negative Tweet Written: " + negTwt.size());
		outPos.flush();
		outNeg.flush();
		outPos.close();
		outNeg.close();
	}

	/**
	 * Read emoticon into a List
	 * @param file
	 * @return
	 * @throws Exception
	 */
	public static List<String> getEmo(File file) throws Exception {
		Vector<String> result = new Vector<String>();
		BufferedReader in = new BufferedReader(new FileReader(file));
		String temp = "";
		while ((temp = in.readLine()) != null)
			result.add(temp);
		in.close();
		return result;
	}
}
