package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

public class Query implements Comparable<Query>{
	String query;
	List<String> words; /* Words with no duplicates and all lower case */
	
	public Query(String query) {
		this.query = new String(query);
		String[] words_array = query.toLowerCase().split(" ");	
		
		
		
		// Use LinkedHashSet to remove duplicates
		words_array = (new LinkedHashSet<String>(Arrays.asList(words_array))).toArray(new String[0]);
		words = new ArrayList<String>(Arrays.asList(words_array));
	}
	
	@Override
	public int compareTo(Query arg0) {
		return this.query.compareTo(arg0.query);
	}
	
	@Override
	public String toString() {
	  return query;
	}
	
	// Handle the query vector
		public Map<String,Double> getQueryFreqs() {
			Map<String,Double> tfQuery = new HashMap<String, Double>(); // queryWord -> term frequency
			for (String term : this.words) {
				if (tfQuery.containsKey(term)) {
					tfQuery.put(term, tfQuery.get(term)+1.0);
				} else {
					tfQuery.put(term, 1.0);
				}
			}
			// Sublinear scaling
			for (String term : tfQuery.keySet()) {
				if (tfQuery.get(term) != 1.0) System.out.println(tfQuery.get(term));
				tfQuery.put(term, Math.log(tfQuery.get(term)) + 1.0);
			}
			return tfQuery;
		}
}
