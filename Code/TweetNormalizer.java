import java.io.*;

/*
 * This class normalizes tweets from the perspective below:
 * 1. Replace usernames into token USERNAME
 * 2. Turn all tweets into Upper Case
 * 3. Normalize all multiple occurrence of a character in word into 2
 * */

public class TweetNormalizer{
	public static void main(String[] args) throws Exception{
		normalize(
		new File(ConstantClass.folder+"posAttack"),
		new File(ConstantClass.folder+"posAttack2")
		);
	}

	public static void normalize(File input, File output) throws Exception{
		BufferedReader in = new BufferedReader(new FileReader(input));
		BufferedWriter out= new BufferedWriter(new FileWriter(output));
		String s="";
		while((s=in.readLine())!=null){
			s=capitalize(s);
			s=username(s);
			s=multiple(s);
			s=s.trim();
			out.write(s+"\n");
		}
		out.flush();
		out.close();
		in.close();
	}

	private static String capitalize(String s){
		return s.toUpperCase();
	}
	private static String username(String s){
		return s.replaceAll("@[a-zA-Z0-9_]+","USERNAME");
	}
	private static String multiple(String s){
		return s.replaceAll("([A-Za-z])\\1{2,}", "$1$1");
	}
}
