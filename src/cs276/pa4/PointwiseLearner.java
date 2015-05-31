package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointwiseLearner extends Learner {	
	private static final Map<String, Integer> FIELD_MAP;
	private boolean bm25, pr, window;
    static {
        Map<String, Integer> map = new HashMap<String, Integer>();
        map.put("url", 0);
        map.put("title", 1);
        map.put("body", 2);
        map.put("header", 3);
        map.put("anchor", 4);
        FIELD_MAP = Collections.unmodifiableMap(map);
    }
    
    public PointwiseLearner(boolean bm25, boolean pr, boolean window) {
    	this.bm25 = bm25;
    	this.pr = pr;
    	this.window = window;
    }

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {		
		/*
		 * @TODO: Your code here
		 */
		Map<Query,List<Document>> train_data; Map<String, Map<String, Double>> rel_data;
		BM25Scorer bm25_scorer;
		SmallestWindowScorer window_scorer;
		try {
			/* query -> documents */
			train_data = Util.loadTrainData(train_data_file);
			/* query -> (url -> score) */
			rel_data = Util.loadRelData(train_rel_file);
			bm25_scorer = new BM25Scorer(idfs, train_data);
			window_scorer = new SmallestWindowScorer(idfs);
		} catch (Exception e) {
			System.err.println("Error while loading training data: " + e);
			return null;
		}
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		attributes.add(new Attribute("bm25_w"));
		attributes.add(new Attribute("pr"));
		attributes.add(new Attribute("window"));
		
		dataset = new Instances("train_dataset", attributes, 0);
		
		for (Query query : train_data.keySet()) {
			Map<String,Double> query_tfs = query.getQueryFreqs();
			for (Document doc : train_data.get(query)) {	
				double[] instance = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				Map<String,Map<String, Double>> tfs = doc.getDocTermFreqs(query);
				for (String field : tfs.keySet()) {
					double score = 0;
					Map<String, Double> field_tfs = tfs.get(field);
					for (String term : query.words) {
						if (idfs.containsKey(term)) {
							score += idfs.get(term)*query_tfs.get(term)*field_tfs.get(term);
						}
					}
					instance[FIELD_MAP.get(field)] = score;
				}
				if (bm25) instance[5] = bm25_scorer.getSimScore(doc, query);
				if (window) instance[6] = window_scorer.getSimScore(doc, query);
				if (pr) instance[7] = doc.page_rank;
				instance[dataset.numAttributes() - 1] = rel_data.get(query.toString()).get(doc.url);
				Instance inst = new DenseInstance(1.0, instance);
				dataset.add(inst);
			}
		}
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 */
		LinearRegression model = new LinearRegression();
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			System.err.println("Error while training linear regression: " + e);
		}
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		Map<Query,List<Document>> test_data;
		BM25Scorer bm25_scorer;
		SmallestWindowScorer window_scorer;
		try {
			/* query -> documents */
			test_data = Util.loadTrainData(test_data_file);
			bm25_scorer = new BM25Scorer(idfs, test_data);
			window_scorer = new SmallestWindowScorer(idfs);
		} catch (Exception e) {
			System.err.println("Error while loading training data: " + e);
			return null;
		}
		
		Instances dataset = null;
		/* {query -> {doc -> index}} */
		Map<String, Map<String, Integer>> index_map = new HashMap<String, Map<String, Integer>>();
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		attributes.add(new Attribute("bm25_w"));
		attributes.add(new Attribute("pr"));
		attributes.add(new Attribute("window"));
		dataset = new Instances("test_dataset", attributes, 0);
		
		int index = 0;
		for (Query query : test_data.keySet()) {
			Map<String,Double> query_tfs = query.getQueryFreqs();
			index_map.put(query.toString(), new HashMap<String, Integer>());
			for (Document doc : test_data.get(query)) {	
				double[] instance = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				Map<String,Map<String, Double>> tfs = doc.getDocTermFreqs(query);
				for (String field : tfs.keySet()) {
					double score = 0;
					Map<String, Double> field_tfs = tfs.get(field);
					for (String term : query.words) {
						if (idfs.containsKey(term)) {
							score += idfs.get(term)*field_tfs.get(term)*query_tfs.get(term);
						}
					}
					instance[FIELD_MAP.get(field)] = score;
				}
				if (bm25) instance[5] = bm25_scorer.getSimScore(doc, query);
				if (window) instance[6] = window_scorer.getSimScore(doc, query);
				if (pr) instance[7] = doc.page_rank;
				Instance inst = new DenseInstance(1.0, instance);
				dataset.add(inst);
				index_map.get(query.toString()).put(doc.url, index);
				index += 1;
			}
		}
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		TestFeatures test_features = new TestFeatures();
		test_features.features = dataset;
		test_features.index_map = index_map;

		return test_features;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		double eta = 0.000000001;
		for (int i = 0; i < 10; i++) {
			System.out.println(((LinearRegression) model).coefficients()[i]);
		}
		Map<String, List<String>> ranked_queries = new HashMap<String, List<String>>();
		Instances test_dataset = tf.features;
		Map<String, Map<String, Integer>> index_map = tf.index_map;
		
		try {
			for (String query : index_map.keySet()) {
				TreeMap<Double, String> scoreMap = new TreeMap<Double, String>();
				ranked_queries.put(query, new ArrayList<String>());
				for (String doc : index_map.get(query).keySet()) {
					int index = index_map.get(query).get(doc);
					// Negate so that order is highest to lowest
					double score = -1*model.classifyInstance(test_dataset.instance(index));
					while (scoreMap.containsKey(score)) {
						score += eta;
					}
					scoreMap.put(score, doc);
				}
				for (int i = 0; i < index_map.get(query).keySet().size(); i++) {
					ranked_queries.get(query).add(scoreMap.get(scoreMap.firstKey()));
					scoreMap.remove(scoreMap.firstKey());
				}
			}
		} catch (Exception e) {
			System.err.println("Error while classifying test: ");
			e.printStackTrace();
		}		
		
		return ranked_queries;
	}

}
