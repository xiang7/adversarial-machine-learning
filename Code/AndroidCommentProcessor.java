import java.io.*;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.util.*;
/**
 * Turn andoid comments into the same format with tweet files.
 * The output format is: emoticon+ + title+ +text.
 * All comments with rating of 4 or 5 is marked with a :). 
 * Those with rating of 1 or 2 is marked with a :(. 
 * Comments with rating of 3 is discarded. The output 
 * of this class is fed into tweet processor for next steps.
 * @author 6
 *
 */
public class AndroidCommentProcessor {

	private static String delimiter="|";
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		
		process(new File(ConstantClass.AndroidComments),
				new File(ConstantClass.folder+"pos"),
				new File(ConstantClass.folder+"neg"),
				4,
				2);
	}
	
	/**
	 * Turn android comment file into a tweet file to be processed later
	 * @param inFile
	 * @param outFile
	 * @param highThd - the rate above or equal to which a comment is considered as positive
	 * @param lowThd - the rate below or equal to which a comment is considered as negative
	 * @throws Exception
	 */
	public static void process(File inFile,File posOutFile,File negOutFile,int highThd,int lowThd) throws Exception{
		Hashtable<String,Integer> table=new Hashtable<String,Integer>();  
		BufferedReader in = new BufferedReader(new FileReader(inFile));
		BufferedWriter posOut=new BufferedWriter(new FileWriter(posOutFile));
		BufferedWriter negOut=new BufferedWriter(new FileWriter(negOutFile));
		String temp="";
		int poscount=0;
		int negcount=0;
		int totalCount=0;
		in.readLine(); //skip the title line
		List<String> posEmo=TweetProcessor.getEmo(new File(ConstantClass.PositiveEmoticon));
		List<String> negEmo=TweetProcessor.getEmo(new File(ConstantClass.NegativeEmoticon));
		while((temp=in.readLine())!=null)
		{
			String[] split=temp.split("\\"+delimiter);
			int score = (int)(Double.valueOf(split[2]).doubleValue());
			if(score>lowThd&&score<highThd)
				continue;
			if(!isPureAscii(temp))
				continue;
			
			String re="";
			if(split.length<=4)
				continue;
			if(split.length==5||split[5].trim().length()==0)
				re+=split[4];
			else
				re+=split[4]+" "+split[5];
			
			re=re.toUpperCase();
			re= re.replaceAll("HTTP://.*? ", "URL ");
			re= re.replaceAll("HTTPS://.*? ", "URL ");
			re= re.trim().replaceAll("([A-Z])\\1{2,}", "$1$1");
			re=re.replaceAll("\\p{Punct}", " ");
			
			if (re.indexOf(":P") != -1)
				continue;
			
			for (String emo : posEmo) {
				if (re.indexOf(emo) != -1) {
					re = re.replaceAll(emo.replaceAll("\\)", "\\\\)"), " ");
				}
			}
			for (String emo : negEmo) {
				if (re.indexOf(emo) != -1) {
					re = re.replaceAll(emo.replaceAll("\\(", "\\\\("), " ");
				}
			}
			
			re=re.trim().replaceAll("\\s+", " ");
			if (re.split("\\s+").length <= 1)
				continue;
			
			if(table.containsKey(re))
				continue;
			

			if(score<=lowThd)
			{
				negcount++;
				table.put(re, ++totalCount);
				re=totalCount+ConstantClass.delimiter+re;
				negOut.write(re+"\n");
			}
		else
			{
			if(poscount<=negcount) {
				table.put(re, ++totalCount);
			re=totalCount+ConstantClass.delimiter+re;
			poscount++;
			posOut.write(re+"\n");
			}
			}
		}
		negOut.flush();
		posOut.flush();
		negOut.close();
		posOut.close();
		in.close();
		System.out.println("Pos: "+poscount);
		System.out.println("Neg: "+negcount);
	}
	
	public static boolean isPureAscii(String v)
	{
		byte[] b=v.getBytes();
		CharsetDecoder d=Charset.forName("US-ASCII").newDecoder();
		try {
			CharBuffer r = d.decode(ByteBuffer.wrap(b));
		}catch(CharacterCodingException e)
		{return false;}
		return true;
		}
}
