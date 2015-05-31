package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {
  private LibSVM model;
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
  public PairwiseLearner(boolean isLinearKernel, boolean bm25, boolean pr, boolean window){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    this.bm25 = bm25;
	this.pr = pr;
	this.window = window;
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel, boolean bm25, boolean pr, boolean window){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    this.bm25 = bm25;
	this.pr = pr;
	this.window = window;
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  	public Instances standardization(Instances X) {
  		try {
	  		Standardize filter = new Standardize();
	  		filter.setInputFormat(X);
	  		Instances new_X = Filter.useFilter(X, filter);
	  		return new_X;
  		} catch (Exception e) {
  			System.err.println("Error standardizing instances" + e);
  			return null;
  		}
  	}
  	
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
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
		
		Instances dataset = null, pairs = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("bm25_w"));
		attributes.add(new Attribute("pr"));
		attributes.add(new Attribute("window"));
		ArrayList<String> labels = new ArrayList<String>();
		labels.add("greater");
		labels.add("lesser");
		attributes.add(new Attribute("relevance_score", labels));
		dataset = new Instances("train_dataset", attributes, 0);
		pairs = new Instances("train_dataset", attributes, 0);
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
							score += idfs.get(term)*field_tfs.get(term)*query_tfs.get(term);
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
			dataset = standardization(dataset);
			// swap classes to maintain even distribution
			boolean current = true;
			for (int i = 0; i < dataset.size(); i++) {
				for (int j = i+1; j < dataset.size(); j++) {
					double[] diff = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
					if (dataset.get(i).equals(dataset.get(j))) continue;
					if (current) {
						diff[8] = 0;
					} else {
						diff[8] = 1;
					}
					if (diff[8] == 0 && dataset.get(i).value(8) >= dataset.get(j).value(8) ||
						diff[8] == 1 && dataset.get(i).value(8) < dataset.get(j).value(8)) {
						for (int k = 0; k < 8; k++) {
							diff[k] = dataset.get(i).value(k) - dataset.get(j).value(k);
						}
					} else {
						for (int k = 0; k < 8; k++) {
							diff[k] = dataset.get(j).value(k) - dataset.get(i).value(k);
						}
					}
					current = !current;
					Instance diff_inst = new DenseInstance(1.0, diff);
					pairs.add(diff_inst);
				}
			}
			dataset.clear();
		}
		/* Set last attribute as target */
		pairs.setClassIndex(pairs.numAttributes() - 1);
		
		return pairs;
	}

	@Override
	public Classifier training(Instances dataset) {
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(dataset.toSummaryString());
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		Learner learner = new PointwiseLearner(false, false, false);
		TestFeatures test_features = learner.extract_test_features(test_data_file, idfs);
		test_features.features = standardization(test_features.features);
		return test_features;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		double eta = 0.000000001;
		Map<String, List<String>> ranked_queries = new HashMap<String, List<String>>();
		Instances test_dataset = tf.features;
		Map<String, Map<String, Integer>> index_map = tf.index_map;
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("bm25_w"));
		attributes.add(new Attribute("pr"));
		attributes.add(new Attribute("window"));
		ArrayList<String> labels = new ArrayList<String>();
		labels.add("greater");
		labels.add("lesser");
		attributes.add(new Attribute("relevance_score", labels));
		try {
			for (String query : index_map.keySet()) {
				TreeMap<Double, String> scoreMap = new TreeMap<Double, String>();
				ranked_queries.put(query, new ArrayList<String>());
				for (String doc1 : index_map.get(query).keySet()) {
					double rank = 0;
					int index1 = index_map.get(query).get(doc1);
					Instance i1 = test_dataset.instance(index1);
					for (String doc2 : index_map.get(query).keySet()) {
						if (doc1.equals(doc2)) continue;
						int index2 = index_map.get(query).get(doc2);
						Instance i2 = test_dataset.instance(index2);
						double[] diff = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};;
						for (int i = 0; i < 8; i++) {
							diff[i] = i1.value(i) - i2.value(i);
						}
						Instance diff_inst = new DenseInstance(1.0, diff);
						Instances data = new Instances("comp_dataset", attributes, 0);
						data.setClassIndex(8);
						diff_inst.setDataset(data);
						if (model.classifyInstance(diff_inst) == 0) rank--;
					}
					while (scoreMap.containsKey(rank)) {
						rank += eta;
					}
					scoreMap.put(rank, doc1);
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
