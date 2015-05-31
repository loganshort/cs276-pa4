package cs276.pa4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Skeleton code for the implementation of a Cosine Similarity Scorer in Task 1.
 */
public class CosineSimilarityScorer extends AScorer {

	public CosineSimilarityScorer(Map<String,Double> idfs) {
		super(idfs);
		weights = new HashMap<String, Double>();
		weights.put("url", urlweight);
		weights.put("title", titleweight);
		weights.put("body", bodyweight);
		weights.put("header", headerweight);
		weights.put("anchor", anchorweight);
	}

	/////////////// Weights //////////////////
	double urlweight = 1;
	double titleweight  = 0.8;
	double bodyweight = 0.6;
	double headerweight = 0.5;
	double anchorweight = 1;
	Map<String, Double> weights;

	double smoothingBodyLength = 1000; // Smoothing factor when the body length is 0.
	
	//////////////////////////////////////////

	public double getNetScore(Map<String, Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery, Document d) {
		double score = 0.0;
		
		// Compute dot product.
		List<Double> tfDocument = new ArrayList<Double>();
		for (String field : tfs.keySet()) {
			Map<String, Double> current = tfs.get(field);
			for (String term : q.words) {
				if (idfs.containsKey(term)) {
					score += idfs.get(term)*tfQuery.get(term)*weights.get(field)*current.get(term);
				}
			}
		}
		return score;
	}

	// Normalize the term frequencies. Note that we should give uniform normalization to all fields as discussed
	// in the assignment handout.
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
		for (Map<String, Double> current : tfs.values()) {
			for (String term : current.keySet()) {
				current.put(term, current.get(term) / (d.body_length + smoothingBodyLength));
			}
		}
	}


	@Override
	public double getSimScore(Document d, Query q) {
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);

	    return getNetScore(tfs,q,tfQuery,d);
	}

}
