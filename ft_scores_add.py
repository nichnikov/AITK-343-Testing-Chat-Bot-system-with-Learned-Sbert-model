import os
import fasttext
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ft_model = fasttext.load_model(os.path.join("models", "bss_cbow_lem_clear.bin"))
ft_model = fasttext.load_model(os.path.join("models", "bss_cbow_lem.bin"))
test_df = pd.read_csv(os.path.join("results", "searching_results_es_230405.csv"), sep="\t")

for cl in test_df:
    print(cl)

unique_queries = list(set(test_df["Query"]))
print(len(unique_queries))

queries_dicts = test_df.to_dict(orient="records")
for num, d in enumerate(queries_dicts):
    print(num)
    q_v = ft_model.get_sentence_vector(d["LemQuery"])
    e_v = ft_model.get_sentence_vector(d["LemEtalon"])
    sc = cosine_similarity(q_v.reshape(1, 100), e_v.reshape(1, 100))
    d["FtSim"] = sc[0][0]

results_with_ft_sim = pd.DataFrame(queries_dicts)
results_with_ft_sim.to_csv(os.path.join("results", "searching_results_texts_ftsim_230405.csv"), sep="\t", index=False)

