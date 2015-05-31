package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Skeleton code for the implementation of a BM25 Scorer in Task 2.
 */
public class BM25Scorer extends AScorer {
	Map<Query,List<Document>> queryDict; // query -> url -> document

	public BM25Scorer(Map<String,Double> idfs, Map<Query,List<Document>> queryDict) {
		super(idfs);
		bvalues = new HashMap<String, Double>();
		bvalues.put("url", burl);
		bvalues.put("title", btitle);
		bvalues.put("header", bheader);
		bvalues.put("body", bbody);
		bvalues.put("anchor", banchor);
		weights = new HashMap<String, Double>();
		weights.put("url", urlweight);
		weights.put("title", titleweight);
		weights.put("body", bodyweight);
		weights.put("header", headerweight);
		weights.put("anchor", anchorweight);
		this.queryDict = queryDict;
		this.calcAverageLengths();
	}


	/////////////// Weights /////////////////
	double urlweight = 1;
	double titleweight  = 1;
	double bodyweight = 0.1;
	double headerweight = 0.3;
	double anchorweight = 0.1;
	Map<String, Double> weights;

	/////// BM25 specific weights ///////////
	double burl=0.75;
	double btitle=0.75;
	double bheader=0.7;
	double bbody=0.75;
	double banchor=0.4;
	Map<String, Double> bvalues;

	double k1=3.0;
	double pageRankLambda=2.0;
	double pageRankLambdaPrime=3.0;
	//////////////////////////////////////////

	/////// BM25 data structures - feel free to modify ///////

	Map<Document,Map<String,Double>> lengths; // Document -> field -> length
	Map<String,Double> avgLengths;  // field name -> average length
	Map<Document,Double> pagerankScores; // Document -> pagerank score

	//////////////////////////////////////////

	// Set up average lengths for bm25, also handles pagerank
	public void calcAverageLengths() {
		lengths = new HashMap<Document,Map<String,Double>>();
		avgLengths = new HashMap<String,Double>();
		pagerankScores = new HashMap<Document,Double>();
		
		//TODO : Your code here
		for (String field : this.TFTYPES) {
			avgLengths.put(field,0.0);
		}
		int numDocuments = 0;
		for (Query query : queryDict.keySet()) {
			numDocuments += queryDict.get(query).size();
			for (Document doc : queryDict.get(query)) {
				Map<String,Double> docDict = new HashMap<String,Double>();
				
				if (doc.title != null) {
					Double urlLength = getLength(doc.url);
					docDict.put("url",urlLength);
					avgLengths.put("url", avgLengths.get("url")+urlLength);
				}
				
				if (doc.title != null) {
					Double titleLength = getLength(doc.title);
					docDict.put("title",titleLength);
					avgLengths.put("title", avgLengths.get("title")+titleLength);
				}
				
				if (doc.headers != null) {
					Double headerLength = getLength(doc.headers);
					docDict.put("header",headerLength);
					avgLengths.put("header", avgLengths.get("header")+headerLength);
				}
				
				if (doc.body_hits != null) {
					Double bodyLength = getLength(doc.body_hits.keySet());
					docDict.put("body",bodyLength);				
					avgLengths.put("body", avgLengths.get("body")+bodyLength);
				}
				
				if (doc.anchors != null) {
					Double anchorLength = getAnchorLength(doc.anchors);
					docDict.put("anchor",anchorLength);
					avgLengths.put("anchor", avgLengths.get("anchor")+anchorLength);
				}
				
				pagerankScores.put(doc,1.0*doc.page_rank);
				lengths.put(doc,docDict);
			}
		}
		
		//normalize avgLengths
		for (String tfType : this.TFTYPES) {
			//TODO : Your code here
			avgLengths.put(tfType, avgLengths.get(tfType)/(1.0*numDocuments));

		}

	}
	
	private double getLength(String input) {
		return 1.0*input.trim().split("\\s+").length;
	}
	
	private double getLength(List<String> input) {
		double length = 0.0;
		for (String s : input) {
			length += 1.0*s.trim().split("\\s+").length;
		}
		return length;
	}
	
	private double getLength(Set<String> input) {
		double length = 0.0;
		for (String s : input) {
			length += 1.0*s.trim().split("\\s+").length;
		}
		return length;
	}
	
	private double getAnchorLength(Map<String, Integer> input) {
		double length = 0.0;
		for (String s : input.keySet()) {
			length += 1.0*s.trim().split("\\s+").length*input.get(s);
		}
		return length;
	}

	////////////////////////////////////


	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d) {
		double score = 0.0;
		
		//TODO : Your code here
		for (String term : q.words) {
			Double wdt = 0.0;
			for (String field : tfs.keySet()) {
				wdt += weights.get(field) * tfs.get(field).get(term);
			}
			if (idfs.containsKey(term)) {
				score += (wdt/(k1 + wdt))*idfs.get(term);
			}
			
		}	
		score += pageRankScore(d);
		return score;
	}
	
	private double pageRankScore(Document d) {
		return pageRankLambda * Math.log(pageRankLambdaPrime+pagerankScores.get(d));
	}
	
	/*private double pageRankScore(Document d) {
		return pageRankLambda * pagerankScores.get(d)/(pageRankLambdaPrime + pagerankScores.get(d));
	}*/
	
	/*private double pageRankScore(Document d) {
		return pageRankLambda * 1/(pageRankLambdaPrime + Math.exp(-1*pagerankScores.get(d)*pageRankLambdaPrime));
	}*/

	//do bm25 normalization
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
		//TODO : Your code here
		for (String field : tfs.keySet()) {
			Map<String,Double> fieldtf = tfs.get(field);
			for (String term : fieldtf.keySet()) {
				Double tf = fieldtf.get(term);
				if (avgLengths.get(field) < 0.000001 && avgLengths.get(field) > -0.000001) {
					fieldtf.put(term, 0.0);
				} else {
					if (lengths.get(d).get(field) == null) {
						fieldtf.put(term, tf/(1 + bvalues.get(field)*((0.0/avgLengths.get(field))-1)));
					} else {
						if(field.equals("url")) {
							if (tf != 0.0)
							System.out.println(tf);
						}
						fieldtf.put(term, tf/(1 + bvalues.get(field)*((lengths.get(d).get(field)/avgLengths.get(field))-1)));
					}
				}
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
