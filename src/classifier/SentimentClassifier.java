package classifier;

import java.io.File;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

abstract public class SentimentClassifier {
	protected FilteredClassifier filteredClassifier;
	
	public Instances prepareTrainingData(String arffFileName) throws Exception {

		DataSource loader = new DataSource(arffFileName); 
		Instances loadedData = loader.getDataSet();
		loadedData.setClassIndex(1);

		return loadedData;
	}
	
	public void buildClassifier(Instances trainingData) throws Exception {
		Filter filter = buildTextToWordVectorFilter(trainingData);
		buildFilteredClassifier(trainingData, filter);
	}
	
	private Filter buildTextToWordVectorFilter(Instances trainingData) throws Exception {
		NGramTokenizer tokenizer = new NGramTokenizer(); 
		tokenizer.setNGramMinSize(1); 
		tokenizer.setNGramMaxSize(2); 
		tokenizer.setDelimiters(" \\W");
		
		StringToWordVector textToWordFilter = new StringToWordVector();
		
		textToWordFilter.setAttributeIndices("first");
		textToWordFilter.setTokenizer(tokenizer);
		textToWordFilter.setWordsToKeep(10000);
		textToWordFilter.setInputFormat(trainingData);
		textToWordFilter.setDoNotOperateOnPerClassBasis(true);
		textToWordFilter.setLowerCaseTokens(true);
		textToWordFilter.setStopwords(new File("stopwords.txt"));
		Ranker ranker = new Ranker();
		ranker.setThreshold(0.0);

		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(new InfoGainAttributeEval());
		asFilter.setSearch(ranker);
		
		Filter[] filters = new Filter[2];
		filters[0] = textToWordFilter;
		filters[1] = asFilter;
		
		MultiFilter multiFilter = new MultiFilter();
		multiFilter.setFilters(filters);
		
		return multiFilter;
	}
	
	abstract protected void buildFilteredClassifier(Instances trainingData, Filter filter) throws Exception;
}
