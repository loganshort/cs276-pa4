package cs276.pa4;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Document {
	public String url = null;
	public String title = null;
	public List<String> headers = null;
	public Map<String, List<Integer>> body_hits = null; // term -> [list of positions]
	public int body_length = 0;
	public int page_rank = 0;
	public Map<String, Integer> anchors = null; // term -> anchor_count

	// For debug
	public String toString() {
		StringBuilder result = new StringBuilder();
		String NEW_LINE = System.getProperty("line.separator");
		if (title != null) result.append("title: " + title + NEW_LINE);
		if (headers != null) result.append("headers: " + headers.toString() + NEW_LINE);
		if (body_hits != null) result.append("body_hits: " + body_hits.toString() + NEW_LINE);
		if (body_length != 0) result.append("body_length: " + body_length + NEW_LINE);
		if (page_rank != 0) result.append("page_rank: " + page_rank + NEW_LINE);
		if (anchors != null) result.append("anchors: " + anchors.toString() + NEW_LINE);
		return result.toString();
	}
	
	/*/
	 * Creates the various kinds of term frequencies (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation.
	 */
	public Map<String,Map<String, Double>> getDocTermFreqs(Query q) {
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
		if (this.url != null) {
			for (String term : this.url.toLowerCase().split("\\.")) {
				if (!terms.contains(term)) continue;
				tfsUrl.put(term, tfsUrl.get(term)+1.0);
			}
		}
		if (this.title != null) {
			for (String term : this.title.toLowerCase().split("\\s+")) {
				if (!terms.contains(term)) continue;
				tfsTitle.put(term, tfsTitle.get(term)+1.0);
			}
		}
		if (this.body_hits != null) {
			for (String term : this.body_hits.keySet()) {
				if (!terms.contains(term)) continue;
				tfsBody.put(term, tfsBody.get(term)+this.body_hits.get(term).size());
			}
		}
		if (this.headers != null) {
			for (String header : this.headers) {
				for (String term : header.split("\\s+")) {
					if (!terms.contains(term)) continue;
					tfsHeader.put(term, tfsHeader.get(term)+1.0);
				}
			}
		}
		if (this.anchors != null) {
			for (String anchor : this.anchors.keySet()) {
				for (String term : anchor.split("\\s+")) {
					if (!terms.contains(term)) continue;
					tfsAnchor.put(term, tfsAnchor.get(term)+(1.0*this.anchors.get(anchor)));
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
