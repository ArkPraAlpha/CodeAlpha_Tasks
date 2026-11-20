"""Topic modelling (LDA) to discover what people talk about in negative texts."""
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def lda_topics(texts, n_topics=4, n_words=8, seed=42, min_docs=40):
    texts = [t for t in texts if t]
    if len(texts) < min_docs:
        return pd.DataFrame()
    vec = CountVectorizer(stop_words="english", min_df=3, max_df=0.8, ngram_range=(1, 2))
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed, learning_method="batch")
    doc_topics = lda.fit_transform(X)
    vocab = vec.get_feature_names_out()
    share = pd.Series(doc_topics.argmax(axis=1)).value_counts(normalize=True)
    rows = []
    for i, comp in enumerate(lda.components_):
        words = [vocab[j] for j in comp.argsort()[::-1][:n_words]]
        rows.append({"topic": i + 1, "top_terms": ", ".join(words),
                     "share_of_texts_pct": round(share.get(i, 0) * 100, 1)})
    return pd.DataFrame(rows)
