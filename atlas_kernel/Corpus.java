import com.fasterxml.jackson.databind.ObjectMapper;
import io.anserini.search.SimpleSearcher;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

class Corpus {
    public static void main(String[] args) throws Exception {
        var json = new ObjectMapper();
        try (var searcher = new SimpleSearcher(args[0]);
             var input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            searcher.set_bm25(0.9f, 0.4f);
            if (searcher.get_total_num_docs() < 95000) throw new IOException("Incomplete BrowseComp-Plus index");
            for (String line; (line = input.readLine()) != null;) {
                var req = json.readTree(line);
                if (req.has("open")) {
                    var record = json.readTree(searcher.doc_raw(req.get("open").asText()));
                    System.out.println(json.writeValueAsString(record.path("contents").asText()));
                    continue;
                }
                var output = new ArrayList<Map<String, String>>();
                for (var hit : searcher.search(req.get("search").asText(), 12)) {
                    var record = json.readTree(searcher.doc_raw(hit.docid));
                    var body = record.path("contents").asText();
                    output.add(Map.of("id", hit.docid, "title", record.path("title").asText(hit.docid),
                                      "snippet", body.substring(0, Math.min(body.length(), 650))));
                }
                System.out.println(json.writeValueAsString(output));
            }
        }
    }
}
