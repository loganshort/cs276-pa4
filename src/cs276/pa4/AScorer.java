package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An abstract class for a scorer. Need to be extended by each specific implementation of scorers.
 */
public abstract class AScorer {
	
	Map<String,Double> idfs; // Map: term -> idf
    // Various types of term frequencies that you will need
	String[] TFTYPES = {"url","title","body","header","anchor"};
	
	public AScorer(Map<String,Double> idfs) {
		this.idfs = idfs;
	}
	
	// Score each document for each query.
	public abstract double getSimScore(Document d, Query q);
	
	// Handle the query vector
	public Map<String,Double> getQueryFreqs(Query q) {
		Map<String,Double> tfQuery = new HashMap<String, Double>(); // queryWord -> term frequency
		for (String term : q.words) {
			if (tfQuery.containsKey(term)) {
				tfQuery.put(term, tfQuery.get(term)+1.0);
			} else {
				tfQuery.put(term, 1.0);
			}
		}
		// Sublinear scaling
		for (String term : tfQuery.keySet()) {
			tfQuery.put(term, Math.log(tfQuery.get(term)) + 1.0);
		}
		return tfQuery;
	}
	
	
	/*/
	 * Creates the various kinds of term frequencies (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation.
	 */
	public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q) {
		// Map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();
		Map<String, Double> tfsUrl = new HashMap<String, Double>();
		Map<String, Double> tfsTitle = new HashMap<String, Double>();
		Map<String, Double> tfsBody = new HashMap<String, Double>();
		Map<String, Double> tfsHeader = new HashMap<String, Double>();
		Map<String, Double> tfsAnchor = new HashMap<String, Double>();
		
		Set<String> terms = new HashSet<String>();
		for (String term : q.words) {
			terms.add(term);
			tfsUrl.put(term, 0.0);
			tfsTitle.put(term, 0.0);
			tfsBody.put(term, 0.0);
			tfsHeader.put(term, 0.0);
			tfsAnchor.put(term, 0.0);
		}
		if (d.url != null) {
			for (String term : d.url.toLowerCase().split("\\s+")) {
				if (!terms.contains(term)) continue;
				tfsUrl.put(term, tfsUrl.get(term)+1.0);
			}
		}
		if (d.title != null) {
			for (String term : d.title.toLowerCase().split("\\s+")) {
				if (!terms.contains(term)) continue;
				tfsTitle.put(term, tfsTitle.get(term)+1.0);
			}
		}
		if (d.body_hits != null) {
			for (String term : d.body_hits.keySet()) {
				if (!terms.contains(term)) continue;
				tfsBody.put(term, tfsBody.get(term)+d.body_hits.get(term).size());
			}
		}
		if (d.headers != null) {
			for (String header : d.headers) {
				for (String term : header.split("\\s+")) {
					if (!terms.contains(term)) continue;
					tfsHeader.put(term, tfsHeader.get(term)+1.0);
				}
			}
		}
		if (d.anchors != null) {
			for (String anchor : d.anchors.keySet()) {
				for (String term : anchor.split("\\s+")) {
					if (!terms.contains(term)) continue;
					tfsAnchor.put(term, tfsAnchor.get(term)+(1.0*d.anchors.get(anchor)));
				}
			}
		}

		tfs.put("url", tfsUrl);
		tfs.put("title", tfsTitle);
		tfs.put("body", tfsBody);
		tfs.put("header", tfsHeader);
		tfs.put("anchor", tfsAnchor);
		
		// Sublinear scaling.
		for (String field : tfs.keySet()) {
			for (String term : tfs.get(field).keySet()) {
				if (tfs.get(field).get(term) > 0) {
					tfs.get(field).put(term, Math.log(tfs.get(field).get(term)) + 1);
				}
			}
		}
		return tfs;
	}

}
