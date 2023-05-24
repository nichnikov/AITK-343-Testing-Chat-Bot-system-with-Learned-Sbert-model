import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v1')
test_df = pd.read_csv(os.path.join("results", "searching_results_es.csv"), sep="\t")

for cl in test_df:
    print(cl)

unique_queries = list(set(test_df["Query"]))
print(len(unique_queries))


queries_dicts = test_df.to_dict(orient="records")
for num, d in enumerate(queries_dicts):
    q_v = vectorizer.encode([str(d["LemQuery"]).lower()])
    e_v = vectorizer.encode([str(d["LemEtalon"]).lower()])
    sc = cosine_similarity(q_v, e_v)
    print(num, sc)
    d["StSim"] = sc[0][0]

results_with_ft_sim = pd.DataFrame(queries_dicts)
results_with_ft_sim.to_csv(os.path.join("results", "searching_results_texts_stsim_lem.csv"), sep="\t", index=False)
