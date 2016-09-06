package classifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;

public class J48Classifier extends SentimentClassifier {

	@Override
	protected void buildFilteredClassifier(Instances trainingData, Filter filter)
			throws Exception {
		J48 classifier = new J48();
		
		filteredClassifier = new FilteredClassifier();
		filteredClassifier.setClassifier(classifier);
		filteredClassifier.setFilter(filter);
		filteredClassifier.buildClassifier(trainingData);
		
		Evaluation eval = new Evaluation(trainingData);
		eval.crossValidateModel(filteredClassifier, trainingData, 10, new Random(1));
		
		System.out.println("J48 RESULTS");
		System.out.println(eval.toSummaryString()); 
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());   
		
	}

}
