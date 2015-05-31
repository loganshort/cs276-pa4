package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 */
public class SmallestWindowScorer extends CosineSimilarityScorer {

	/////// Smallest window specific hyper-parameters ////////
	double B = 20;    	    
	double boostmod = -1;

	//////////////////////////////
	
	public SmallestWindowScorer(Map<String, Double> idfs) {
		super(idfs);
	}
	
	public int findSmallestWindow(List<String> terms, Map<String, Double> q) {
		int smallest = Integer.MAX_VALUE;
		int n = q.keySet().size();
		Set<String> inWindow = new HashSet<String>();
		int start = 0, end = 0;
		for (int i = 0; i < terms.size() - n + 1; i++) {
			if (q.containsKey(terms.get(i))) {
				start = i;
				end = Math.max(end, start+1);
				inWindow.add(terms.get(i));
				while (end < terms.size() && inWindow.size() < n) {
					String current = terms.get(end++);
					if (q.containsKey(current)) {
						inWindow.add(current);
					}
				}
				if (inWindow.size() == n) {
					smallest = Math.min(smallest, end - start + 1);
				}
			}
			inWindow.remove(terms.get(i++));
		}
		return smallest;
	}
	
	public int findSmallestBodyWindow(Map<String, List<Integer>> hits, Map<String, Double> q) {
		int maxPos = 0;
		for (String field : hits.keySet()) {
			for (Integer i : hits.get(field)) {
				maxPos = Math.max(maxPos, i);
			}
		}
		List<String> terms = new ArrayList<String> ();
		for (int i = 0; i < maxPos+1; i++) {
			terms.add("**NotAQueryTerm**");
		}
		for (String field : hits.keySet()) {
			for (Integer i : hits.get(field)) {
				terms.set(i, field);
			}
		}
		return findSmallestWindow(terms, q);
	}
	
	public double getWindow(Document d, Map<String,Double> q) {
		double boost = 1.0;
		int smallest = Integer.MAX_VALUE;
		if (d.url != null) {
			List<String> terms = new ArrayList<String>();
			for (String term : d.url.toLowerCase().split("\\s+")) {
				terms.add(term);
			}
			int current = findSmallestWindow(terms, q);
			if (current < smallest) smallest = current;
		}
		if (d.title != null) {
			List<String> terms = new ArrayList<String>();
			for (String term : d.title.toLowerCase().split("\\s+")) {
				terms.add(term);
			}
			int current = findSmallestWindow(terms, q);
			if (current < smallest) smallest = current;
		}
		if (d.body_hits != null) {
			Map<String, List<Integer>> queryHits = new HashMap<String, List<Integer>>();
			boolean inf = true;
			for (String term : q.keySet()) {
				if (!d.body_hits.containsKey(term) || d.body_hits.get(term).size() == 0) inf = true;
				queryHits.put(term, d.body_hits.get(term));
			}
			if (!inf) {
				int current = findSmallestBodyWindow(queryHits, q);
				if (current < smallest) smallest = current;
			}
		}
		if (d.headers != null) {
			for (String header : d.headers) {
				List<String> terms = new ArrayList<String>();
				for (String term : header.split("\\s+")) {
					terms.add(term);
				}
				int current = findSmallestWindow(terms, q);
				if (current < smallest) smallest = current;
			}
		}
		if (d.anchors != null) {
			for (String anchor : d.anchors.keySet()) {
				List<String> terms = new ArrayList<String>();
				for (String term : anchor.split("\\s+")) {
					terms.add(term);
				}
				int current = findSmallestWindow(terms, q);
				if (current < smallest) smallest = current;
			}
		}
		return smallest;
	}
	
	@Override
	public double getSimScore(Document d, Query q) {
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);
		
		return getWindow(d, tfQuery);
	}

}
