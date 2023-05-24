"""https://elasticsearch-py.readthedocs.io/en/latest/async.html"""
from __future__ import annotations

import os
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from src.config import (logger,
                        PROJECT_ROOT_DIR,
                        parameters)
from src.utils import jaccard_similarity
from src.data_types import TextsDeleteSample
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        env_file = os.path.join(PROJECT_ROOT_DIR, ".env")
        env_file_encoding = "utf-8"

'''
class ElasticSettings(Settings):
    """Elasticsearch settings."""

    hosts: str
    user_name: str | None
    password: str | None

    max_hits: int = parameters.max_hits
    chunk_size: int = parameters.chunk_size

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Returns basic auth tuple if user and password are specified."""
        # print(self.user_name, self.password)
        if self.user_name and self.password:
            return self.user_name, self.password
        return None

setting = ElasticSettings()
'''


class ElasticClient(AsyncElasticsearch):
    """Handling with AsyncElasticsearch"""
    def __init__(self, *args, **kwargs):
        # self.settings = ElasticSettings()
        super().__init__(
            hosts="http://srv01.nlp.dev.msk2.sl.amedia.tech:9200",
            basic_auth=("elastic", "changeme"),
            request_timeout=100,
            max_retries=50,
            retry_on_timeout=True,
            *args,
            **kwargs,
        )
    '''
    super().__init__(
        hosts=os.environ("HOSTS"),
        basic_auth=(os.environ("USER_NAME"), os.environ("PASSWORD")),
        request_timeout=100,
        max_retries=10,
        retry_on_timeout=True,
    )'''

    async def texts_search(self, index: str, searching_field: str, texts: list[str]) -> list:
        async def search(tx: str, field: str):
            resp = await self.search(
                allow_partial_search_results=True,
                min_score=0,
                index=index,
                query={"match": {field: tx}},
                # size=self.settings.max_hits,
                size=100,
            )
            return resp

        texts_search_result = []
        for txt in texts:
            res = await search(txt, searching_field)
            if res["hits"]["hits"]:
                texts_search_result.append(
                    {
                        "text": txt,
                        "search_results": [
                            {
                                **d["_source"],
                                **{"id": d["_id"]},
                                **{"score": d["_score"]},
                            }
                            for d in res["hits"]["hits"]
                        ],
                    }
                )
        return texts_search_result

    async def answer_search(self, index: str, fa_id: int, pub_id: int):
        """Отдельный метод для точного поиска по двум полям"""

        async def fa_search(templateId: int, pubId: int):
            resp = await self.search(
                allow_partial_search_results=True,
                min_score=0,
                index=index,
                query={
                    "bool": {
                        "must": [
                            {"match_phrase": {"templateId": templateId}},
                            {"match_phrase": {"pubId": pubId}},
                        ]
                    }
                },
                size=self.settings.max_hits,
            )
            return resp

        res = await fa_search(fa_id, pub_id)
        if [res["hits"]["hits"]]:
            return {
                "search_results": [
                    {**d["_source"], **{"id": d["_id"]}, **{"score": d["_score"]}}
                    for d in res["hits"]["hits"]
                ]
            }

    async def search_by_field_exactly(
            self, index: str, field_name: str, searched_value
    ):
        """ """
        resp = await self.search(
            allow_partial_search_results=True,
            min_score=0,
            index=index,
            query={"match_phrase": {field_name: searched_value}},
            size=self.settings.max_hits,
        )
        return resp

    async def delete_by_ids(self, index_name: str, del_ids: list):
        """
        :param index_name:
        :param del_ids:
        """
        _gen = ({"_op_type": "delete", "_index": index_name, "_id": i} for i in del_ids)
        await async_bulk(
            self,
            _gen,
            chunk_size=self.settings.chunk_size,
            raise_on_error=False,
            stats_only=True,
        )

    async def delete_by_texts(self, data_for_delete: TextsDeleteSample):
        """
        Перед удалением тексты (входящие и найденные) проверяются на близость по Жаккарду
        удаляются те, которые "ближе", чем скор в DeleteSample
        :param data_for_delete:
        """
        searching_results = self.texts_search(data_for_delete.Index, data_for_delete)

        ids_for_del = []
        for d in searching_results:
            for s_r in d["search_results"]:
                jaccard_coeff = jaccard_similarity(
                    s_r[data_for_delete.FieldName], d["text"]
                )
                if jaccard_coeff >= data_for_delete.Score:
                    ids_for_del.append(s_r["id"])

        await self.delete_by_ids(data_for_delete.Index, ids_for_del)

    async def delete_in_field(self, index: str, field: str, values: list):
        """
        Удаление по точному совпадению со значениями в указанном поле
        """

        async def get_ids(index, field, values):
            ids = []
            for value in values:
                res = await self.search_by_field_exactly(index, field, value)
                ids += [d["_id"] for d in res["hits"]["hits"]]
            return ids

        ids_for_del = await get_ids(index, field, values)
        await self.delete_by_ids(index, ids_for_del)

    async def add_docs(self, index_name: str, docs: list[dict]):
        """
        :param index_name:
        :param docs:
        """
        try:
            _gen = ({"_index": index_name, "_source": doc} for doc in docs)
            await async_bulk(
                self, _gen, chunk_size=self.settings.chunk_size, stats_only=True
            )
            logger.info("adding {} documents in index {}".format(len(docs), index_name))
        except Exception:
            logger.exception(
                "Impossible adding {} documents in index {}".format(
                    len(docs), index_name
                )
            )
