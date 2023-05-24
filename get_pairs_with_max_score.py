import os
import pandas as pd
from collections import namedtuple
from itertools import groupby

ROW = namedtuple("ROW", "Query LemQuery FastAnswID Etalon LemEtalon El_score FtSim")

# df = pd.read_csv(os.path.join("results", "searching_results_texts_ftsim.csv"), sep="\t")
# df = pd.read_csv(os.path.join("results", "searching_results_texts_stsim_lem.csv"), sep="\t")
df = pd.read_csv(os.path.join("results", "searching_results_texts_ftsim_230405.csv"), sep="\t")

# https://stackoverflow.com/questions/9758450/pandas-convert-dataframe-to-array-of-tuples
tpls = list(df.itertuples(index=False))
print(tpls[:10])

for cl in df:
    print(cl)

grp_data = groupby(sorted(tpls, key=lambda x: x.Query), key=lambda y: y.Query)

results = []
for k, v in grp_data:
    d_max = sorted([x for x in v], key=lambda y: y.FtSim, reverse=True)[0]
    # d_max = sorted([x for x in v], key=lambda y: y.StSim, reverse=True)[0]
    results.append(d_max)

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(os.path.join("results", "results_max_ftsim_lem_230405.csv"), sep="\t", index=False)