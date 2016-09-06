package test;

import classifier.J48Classifier;
import classifier.NaiveBayesClassifier;
import classifier.SentimentClassifier;
import preparation.CreateDatasets;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Test {

	public static void main(String[] args) {
		String filePath = "data/reviews.json";
		String trainingArffFile = "reviewsDataset.arff";
		
		CreateDatasets.createDatasets(filePath);
		NaiveBayesClassifier nbClassifier = new NaiveBayesClassifier();
		J48Classifier jClassifier = new J48Classifier();
		
		try {
			nbClassifier.buildClassifier(nbClassifier.prepareTrainingData(trainingArffFile));
			jClassifier.buildClassifier(jClassifier.prepareTrainingData(trainingArffFile));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		

	}

}
