
public class ConstantClass {

	public static int EgyptianDataSet=0;
	public static int ElectionDataSet=1;
	public static int AndoidDataSet=2;
	public static String delimiter="\t";
	public static int dataSetName=ConstantClass.EgyptianDataSet;
	private static String[] classifierNames=new String[] {"SVM","NAIVEBAYES","ADABOOST"};
	public static int SVM=0;
	public static int NAIVEBAYES=1;
	public static int ADABOOST=2;
	public static int currClassifier=ConstantClass.SVM;
	public static String currClassifierName=classifierNames[currClassifier];
	private static String MainFolder="../Data/";
	private static String EgyptianFolder=MainFolder+"Egyptian/";
	private static String ElectionFolder=MainFolder+"Election/";
	private static String AndroidFolder=MainFolder+"Android/";
	private static String[] folderArray=new String[] {EgyptianFolder,ElectionFolder,AndroidFolder};
	public static String folder=folderArray[dataSetName];
	public static String EgyptianTweets=folder+"tweet";
	public static String ElectionTweets=folder+"tweet";
	public static String AndroidComments=folder+"Android.Comments";
	public static String PositiveEmoticon=MainFolder+"PositiveEmoticon";
	public static String NegativeEmoticon=MainFolder+"NegativeEmoticon";
	public static String tagger="I:\\documents\\document\\Purdue\\Research\\NamedEntityRecognition\\Data File\\english-left3words-distsim.tagger";
	public static String FeatureFile=folder+"Features";
	public static String TestFile=MainFolder+"labeled_test";
	private static String tweetTestVec=folder+"TestVec";
	private static String commentTestVec=MainFolder+"Android/TestVec";
	private static String[] testVecArray=new String[] {tweetTestVec,tweetTestVec,commentTestVec};
	public static String TestVec=testVecArray[dataSetName];
	public static int MaxNumFea=254723;
	private static int EgyptianSize=61220;
	private static int ElectionSize=12000;
	private static int AndroidSize=60000;
	private static int[] sizeArray=new int[] {EgyptianSize, ElectionSize,AndroidSize};
	public static int DataSetSize=sizeArray[dataSetName];
}


