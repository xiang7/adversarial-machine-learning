import weka.classifiers.bayes.NaiveBayesMultinomial;
public class MyNaiveBayesMultinomial extends NaiveBayesMultinomial{

	public MyNaiveBayesMultinomial()
	{
		super();
	}
	
	public double[][] getDistribution()
	{
		System.out.println(this.m_numAttributes);
		return this.m_probOfWordGivenClass;
	}
	
	public double[] getClassProb()
	{
		return this.m_probOfClass;
	}
}
