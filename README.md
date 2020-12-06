# Podcast Segment Retrieval Spotify

#### Supervisor: [Judith Bütepage](http://www.csc.kth.se/~butepage/)
##### Collaborators: [Fredrik Segerhammar](https://github.com/Freesegways), [Mariya Lazarova](https://github.com/mimilazarova), [Tianzong Wang](https://github.com/tianzongw)


Locating the best matching paragraph in a document given a search query is a very well studied problem. However, for podcast data the problem is newer and there is not much research done on it. We attempt to retrieve the best jump-in point for relevant segments of podcast episodes given arbitrary user search queries, using the [dataset](https://podcastsdataset.byspotify.com/) provided in the [TREC 2020 Podcasts Track](https://www.aclweb.org/portal/content/trec-2020-podcasts-track-guidelines). We propose two methods, one based traditional statistical methods utilizaing **TF-IDF** and **Okapi BM25**, and another [**Sentence-Transformer**](https://www.sbert.net/index.html) based deep learning embedding method, to target the first Ad-hoc Segment Retrieval task. A detailed project report and presentation will be released later, or upon request.
