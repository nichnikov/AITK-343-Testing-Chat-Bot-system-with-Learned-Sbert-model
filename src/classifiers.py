"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
from itertools import chain
from src.data_types import Parameters
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer
from src.utils import timeout, jaccard_similarity
from src.config import logger
import torch

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

tmt = float(10)  # timeout


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, tokenizer: TextsTokenizer, parameters: Parameters, model, transformer_tokenizer):
        self.es = ElasticClient()
        self.tkz = tokenizer
        self.prm = parameters
        self.model = model
        self.transformer_tokenizer = transformer_tokenizer

    def transformer_classifier(self, query: str, answers: list[str]):
        def predict(premise, hypothesis):
            encoded = self.transformer_tokenizer(premise, hypothesis, return_tensors="pt")
            logits = self.model(**encoded)[0]
            pred_class = torch.argmax(logits).item()
            if pred_class == 1:
                score = max(torch.sigmoid(logits))[0].item()
            else:
                score = 1 - max(torch.sigmoid(logits))[0].item()
            return score
    
    @timeout(float(tmt))
    async def searching(self, text: str, pubid: int, score: float):
        """searching etalon by  incoming text"""
        try:
            tokens = self.tkz([text])
            # print("tokens\n", tokens)
            if tokens[0]:
                tokens_str = " ".join(tokens[0])
                etalons_search_result = await self.es.texts_search(self.prm.clusters_index, "LemCluster", [tokens_str])
                # print("etalons_search_result:\n", [d["Cluster"] for d in etalons_search_result["search_results"]])
                search_results = [[(x["ID"], x["Cluster"], x["ParentPubList"]) for x 
                                                    in d["search_results"]] for d in etalons_search_result]
                ids_clusters_pubs = [tpl for tpl in chain(*search_results)]
                if ids_clusters_pubs:
                    clusters_ids = [(cluster, i, jaccard_similarity(tokens_str, cluster)) for i, cluster, pubs in 
                                    ids_clusters_pubs if jaccard_similarity(tokens_str, cluster) >= score and pubid in pubs]
                    return sorted(list(set(clusters_ids)), key=lambda x: x[2], reverse=True)
                else:
                    logger.info("es didn't find anything for text of tokens {}".format(str(tokens_str)))
                    return [()]
            else:
                logger.info("tokenizer returned empty value for input text {}".format(str(text)))
                return [()]
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return [()]

if __name__ == "__main__":
    import os
    import time
    import pandas as pd
    from src.config import PROJECT_ROOT_DIR, logger

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
