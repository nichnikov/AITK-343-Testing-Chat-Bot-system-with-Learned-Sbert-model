import os
import time
import pandas as pd
import asyncio
from src.start import classifier


test_qrs_df = pd.read_csv(os.path.join("data", "queries_testing.csv"), sep="\t")
result_file_name = "searching_results_es_230414.csv"

test_qrs_dicts = test_qrs_df.to_dict(orient="records")
print(test_qrs_dicts[:10])

loop = asyncio.new_event_loop()

for d in test_qrs_dicts[:5]:
    query = d["Query"]
    sr = loop.run_until_complete(classifier.searching(query, 9, 0.3))
    print("query:", query, "Searching Results:", sr)


"""
results = []
step = 1000
for num, d in enumerate(list(test_qrs_dicts)[:1000]):
    t = time.time()
    result = loop.run_until_complete(classifier.searching_with_transformer(d["Query"]))
    result_ = [(d["Query"], ) + tpl for tpl in result]
    results += result_
    print(num, time.time() - t)
    if num >= step:
        step += 1000
        results_df = pd.DataFrame(results, columns=["Query", "LemQuery", "SysID", "FastAnswID", "DocName", "FastAnswer",
                                                    "ElScore", "TrfScore"])
        results_df.to_csv(os.path.join("results", result_file_name), sep="\t")

loop.close()
results_df = pd.DataFrame(results, columns=["Query", "LemQuery", "SysID", "FastAnswID", "DocName", "FastAnswer",
                                            "ElScore", "TrfScore"])
print(results_df)
results_df.to_csv(os.path.join("results", result_file_name), sep="\t")
"""