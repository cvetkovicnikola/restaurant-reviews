package classifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.Filter;

public class NaiveBayesClassifier extends SentimentClassifier {

	@Override
	protected void buildFilteredClassifier(Instances trainingData, Filter filter)
			throws Exception {
		
		NaiveBayes nbClassifier = new NaiveBayes();
		nbClassifier.setUseSupervisedDiscretization(true);
		
		filteredClassifier = new FilteredClassifier();
		filteredClassifier.setClassifier(nbClassifier);
		filteredClassifier.setFilter(filter);
		filteredClassifier.buildClassifier(trainingData);
		
		Evaluation eval = new Evaluation(trainingData); 
		eval.crossValidateModel(filteredClassifier, trainingData, 10, new Random(1));

		System.out.println("NAIVE BAYES RESULTS");
		System.out.println(eval.toSummaryString()); 
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());   
		
		
	}
	
}
