"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
классификатор ищет по текстам коротких ответов ответам 
"""
import re
import numpy as np
import torch
from itertools import chain
from src.config import logger
from src.utils import timeout
from src.my_parser import TextParser
from src.data_types import Parameters
from src.storage import ElasticClient
from gensim.matutils import sparse2full
from src.texts_processing import TextsTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

tmt = float(300)  # timeout


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, parser: TextParser, parameters: Parameters, model, tokenizer):
        self.es = ElasticClient()
        self.parser = parser
        self.prm = parameters
        self.model = model
        self.tokenizer = tokenizer

    def transformer_classifier(self, query: str, answers: list[str]):
        def predict(premise, hypothesis):
            encoded = self.tokenizer(premise, hypothesis, return_tensors="pt")
            logits = self.model(**encoded)[0]
            pred_class = torch.argmax(logits).item()
            if pred_class == 1:
                score = max(torch.sigmoid(logits))[0].item()
            else:
                score = 1 - max(torch.sigmoid(logits))[0].item()
            return score

        results = []
        for answer in answers:
            sc = predict(query, answer)
            results.append(sc)
        return results

    async def searching_with_transformer(self, text: str):
        """"""
        """searching etalon by  incoming text"""
        try:
            q_pars_dict = self.parser(text)
            etalons_search_result = await self.es.texts_search(self.prm.answers_index, "LemAnswerDocNameText",
                                                               [q_pars_dict["lem_clear_text"]])
            results_tuples = [(q_pars_dict["lem_clear_text"], d["SysID"], d["ID"], d["DocName"],
                               re.sub(r"\\xa0", " ", d["ShortAnswerText"]),
                               d["score"]) for d in etalons_search_result[0]["search_results"] if d["SysID"] == 1]
            searched_answers = [str(answ[4]) for answ in results_tuples[:5]]
            scores = self.transformer_classifier(str(text), searched_answers)
            return [(tpl) + (sc, ) for tpl, sc in zip(results_tuples, scores)]
        except:
            return [("N", 0, 0, "N", "N", 0)]

    @timeout(float(tmt))
    def searching(self, num: int, text: str):
        """"""
        """searching etalon by  incoming text"""
        try:
            item_detales = {"Query": text}
            q_pars_dict = self.parser(text)
            etalons_search_result = self.es.texts_search(self.prm.answers_index, "LemAnswerDocNameText",
                                                         [q_pars_dict["lem_clear_text"]])

            item_detales["QueryParsing"] = q_pars_dict

            results_tuples = [(d["ID"], d["Cluster"], d["LemCluster"]) for d in
                              etalons_search_result[0]["search_results"]]
            if results_tuples:
                q_ft_vc, q_vc, q_bow = self.text2vector(q_pars_dict)
                item_detales["QueryBow"] = str(q_bow)

                ids, ets, lm_ets = zip(*results_tuples)
                ets_pars_dicts = [self.parser(lm_et) for lm_et in list(lm_ets)]

                item_detales["EtalonsParsing"] = ets_pars_dicts[:10]

                et_ft_vcs, et_vcs, et_bows = zip(*[(self.text2vector(et_pars_dict)) for et_pars_dict in ets_pars_dicts])
                item_detales["EtalonsBows"] = str(et_bows[:10])

                ft_scores = cosine_similarity(q_ft_vc.reshape(1, 100), et_ft_vcs)[0]
                item_detales["FasTextScore"] = str(list(ft_scores[:10]))

                sorted_vects_ets = [(et_vc, et) for ft_sc, et_vc, et in zip(ft_scores, et_vcs, ets) if ft_sc > 0.9]
                # et_vcs_sort, ets_ = zip(*[(et_vc, et) for ft_sc,
                # et_vc, et in zip(ft_scores, et_vcs, ets) if ft_sc > 0.9])

                if sorted_vects_ets:
                    et_vcs_sort, ets_ = zip(*sorted_vects_ets)
                    et_vcs_ = [v.reshape(1, 100) for v in et_vcs_sort]
                    et_vcs_ = np.concatenate(et_vcs_, axis=0)
                    scores = cosine_similarity(q_vc.reshape(1, 100), et_vcs_)[0]

                    item_detales["Scores"] = str(list(scores[:30]))

                    sorted_search = sorted(list(zip(ids, ets_, "No", scores)), key=lambda x: x[3], reverse=True)
                    print(item_detales)
                    det_fn = "".join([str(num), '.json'])
                    with open(os.path.join(PROJECT_ROOT_DIR, "results", "details", det_fn), "w", encoding='utf8') as jf:
                        json.dump(item_detales, jf, ensure_ascii=False)

                    return sorted_search[0]
                else:
                    return tuple((0, "no", "no", 0))
            else:
                return tuple((0, "no", "no", 0))

        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return tuple((0, "no", "no", 0))


if __name__ == "__main__":
    import os
    import time
    import pandas as pd
    from src.config import PROJECT_ROOT_DIR

    t = time.time()
    tknzr = TextsTokenizer()
    stopwords = []
    stopwords_roots = [os.path.join(PROJECT_ROOT_DIR, "data", "greetings.csv"),
                       os.path.join(PROJECT_ROOT_DIR, "data", "stopwords.csv")]

    for root in stopwords_roots:
        stopwords_df = pd.read_csv(root, sep="\t")
        stopwords += list(stopwords_df["text"])
    tknzr.add_stopwords(stopwords)
    print("TextsTokenizer upload:", time.time() - t)

    t0 = time.time()
    c = FastAnswerClassifier(tknzr)
    print("FastAnswerClassifier upload:", time.time() - t0)

    t1 = time.time()
    r = c.searching("как вернули госпошлины по решение судов", 6, 0.95)
    print("searching time:", time.time() - t1)
    print(r)

    t2 = time.time()
    r = c.searching("электрическая электростанция, чебуркша", 6, 0.95)
    print("searching time:", time.time() - t2)
    print(r)

    print("all working time:", time.time() - t)
