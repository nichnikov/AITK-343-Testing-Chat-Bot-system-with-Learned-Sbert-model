import re
import os
import pandas as pd
from utils import (timeit,
                   timeout)
from pullenti_wrapper.processor import (
    Processor,
    DATE,
    MONEY)
from pullenti_wrapper import referent
from itertools import chain
from collections import namedtuple
from src.texts_processing import TextsTokenizer
from src.config import PROJECT_ROOT_DIR

# pullemti https://www.pullenti.ru/Document
# set of parsers https://habr.com/ru/post/502366/
# рубли https://docs.google.com/spreadsheets/d/1uVllieSyjvu2ElubsFQTBGr_9RtHZVzFOEaGHjog0XU/edit#gid=0


class TextParser:
    def __init__(self, patterns: {}, tokenizer: TextsTokenizer):
        self.patterns = patterns
        self.processor = Processor([DATE, MONEY])
        self.tokenizer = tokenizer

    @timeout(1)
    def pullenti_paser(self, text: str) -> {}:
        """обработка методами библиотеки pullenti"""
        text_parse_dict = {"DATES": [],
                           "MONEY": []}
        DateMoney = namedtuple("DateMoney", "data, spans")
        result = self.processor(text)

        for m in result.matches:
            if type(m.referent) == referent.MoneyReferent:
                text_parse_dict["MONEY"].append(DateMoney((str(m.referent.raw),), m.span))
            if type(m.referent) == referent.DateRangeReferent:
                text_parse_dict["DATES"].append(DateMoney((str(m.referent.raw),), m.span))
            if type(m.referent) == referent.DateReferent:
                year = "".join(["Y", str(m.referent.year)])
                month = "".join(["M", str(m.referent.month)])
                day = "".join(["D", str(m.referent.day)])
                text_parse_dict["DATES"].append(DateMoney((year, month, day), m.span))

        if text_parse_dict["MONEY"]:
            money_patterns = re.compile(r"|".join([text[obj.spans.start:obj.spans.stop]
                                                   for obj in text_parse_dict["MONEY"]]))
            clear_text = money_patterns.sub("MONEY", text)
        else:
            clear_text = text

        if text_parse_dict["DATES"]:
            date_patterns = re.compile(r"|".join([text[obj.spans.start:obj.spans.stop]
                                                  for obj in text_parse_dict["DATES"]]))
            clear_text = date_patterns.sub("DATE", clear_text)
        else:
            clear_text = text

        return {"pullenti_text": text,
                "MONEY": [y for y in chain(*[x.data for x in text_parse_dict['MONEY']])],
                "DATES": [y for y in chain(*[x.data for x in text_parse_dict['DATES']])],
                "pullenti_clear_text": clear_text
                }

    @timeout(1)
    def re_parser(self, text: str, pattern_types: [str]) -> {}:
        """функция парсит входящий текст, извлекая из него паттерны из patterns"""

        text_parse_dict = {"re_text": text}

        for type in self.patterns:
            if type in pattern_types:
                text_parse_dict[type] = self.patterns[type]["pattern"].findall(str(text))
                if "mask" in self.patterns[type]:
                    if text_parse_dict[type]:
                        for pt in text_parse_dict[type]:
                            text = re.sub(pt, self.patterns[type]["mask"], str(text))

                text_parse_dict[type] = [re.sub(r'[^\d\w\s,.\-/\\]', "", t) for t in text_parse_dict[type]]

                if "find_change" in self.patterns[type]:
                    for prt, msk in self.patterns[type]["find_change"]:
                        text_parse_dict[type] = [prt.sub(msk, t) for t in text_parse_dict[type]]

                if "prefix" in self.patterns[type]:
                    prfx = str(self.patterns[type]["prefix"])
                    text_parse_dict[type] = ["".join([prfx, t]) for t in text_parse_dict[type]]

        text_parse_dict["re_clear_text"] = text
        return text_parse_dict

    def text_parser(self, text: str, only_tokenizer: True):
        """
        pipline объединяет все методы поиска
        """
        text_parse_dict = {"text": str(text)}
        tx = str(text).lower()
        if only_tokenizer:
            lem_text_list = self.tokenizer([tx])
            text_parse_dict["lem_clear_text"] = " ".join(lem_text_list[0])
            return text_parse_dict
        else:
            """Предварительная обработка:"""
            tx = re.sub(r'(?:\d{25,})+', "LONGNUMBER", tx)
            pullenti_parse_dict = self.pullenti_paser(tx)
            re_parse_dict_no_lem = self.re_parser(pullenti_parse_dict["pullenti_clear_text"], ["FSBU", "NK", "FZ",
                                                                                               "KBK", "FORMS", "NPA"])
            text_parse_dict["clear_text"] = re_parse_dict_no_lem["re_clear_text"]
            lem_text_list = self.tokenizer([text_parse_dict["clear_text"]])
            re_parse_dict_lem = self.re_parser(" ".join(lem_text_list[0]), ["KEYS", "UNCNOWN_NUM"])
            text_parse_dict["lem_clear_text"] = " ".join(lem_text_list[0])
            del pullenti_parse_dict["pullenti_text"]
            del pullenti_parse_dict["pullenti_clear_text"]
            del re_parse_dict_lem["re_text"]
            del re_parse_dict_lem["re_clear_text"]
            del re_parse_dict_no_lem["re_text"]
            del re_parse_dict_no_lem["re_clear_text"]
            return {**text_parse_dict, **pullenti_parse_dict, **re_parse_dict_no_lem, **re_parse_dict_lem}

    def __call__(self, tx: str):
        return self.text_parser(tx, True)


fsbu_patterns = re.compile(r'фсбу\s*\d+(?:\/|\-)\d+|\d+(?:\/|\-)\d+\s*фсбу')

nk_patterns = re.compile(r'(?:\bнк.{0,1}\b.{0,5}\bст[атьия.]{0,4}|\bст[атьия.]{0,4}\b.{0,5}\bнк|\bпп.{0,1}'
                         r'\b.{0,5}\bп[\.\s]{0,1}\b.{0,5}\bст[атьия.]{0,4}\b.{0,5}\bнк|'
                         r'\bп[\.\s]{0,1}\b.{0,5}\bст[атьия.]{0,4}\b.{0,5}\bнк|'
                         r'ст[атьия.]{0,4}[\s.]{0,3}\d+[\s.]{0,3}\bнк)')

fz_patterns = re.compile(r'(?:\d+.{0,2}фз|фз.{0,2}\d+)')

kbk_patterns = re.compile(r'(?:\bкбк\b|\b\d{10,25}\b)+')

forms_patterns = re.compile(r'(?:\d+.{0,2}\bформ[уыае]\w+|\bформ[уыае].{0,2}\w+\d+|\bформ[уыае]+.{0,2}\w+[-/\\\s+]\d+|'
                            r'\b\w+[-/\\\s+]\d+.{0,2}\bформ[уыае].{0,2}|\d+.{0,2}кнд|кнд.{0,2}\d+)')

ndfl_patterns = re.compile(r'(?:\d\b.{0,2}ндфл|ндфл.{0,2}\d\b|\bндфл\d\b|\dндфл\b)')

npa_patterns = re.compile(r'\d+(?:[-/]\d+)+')

uncknow_num_patterns = re.compile(r'(?:\d+)')

keys_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", "keywords.csv"), sep="\t")
keys_patterns_from_list = re.compile("|".join(list(keys_df["keys"])))
keys_patterns = re.compile(ndfl_patterns.pattern + "|(?:" + keys_patterns_from_list.pattern + ")")

patterns_dict = {
    "FSBU": {"pattern": fsbu_patterns, "mask": " FSBU ",
             "find_change": [(re.compile(r'фсбу|\s'), ""), (re.compile(r'-'), "/")],
             "prefix": "fsbu"},
    "NK": {"pattern": nk_patterns, "mask": " NK ",
           "find_change": [(re.compile(r'нк|п.{0,3}\d+|[\s.-]|[аьият.]'), "")]},
    "FZ": {"pattern": fz_patterns, "mask": " FZ ",
           "find_change": [(re.compile(r'[-./\s]|фз'), "")],
           "prefix": "fz"},
    "KBK": {"pattern": kbk_patterns, "mask": " KBK "},
    "FORMS": {"pattern": forms_patterns, "mask": " FORMA ",
              "find_change": [(re.compile(r'\bформ[уыае]{0,3}|[\s-]'), "")]},
    "NPA": {"pattern": npa_patterns, "mask": " NPA "},
    "KEYS": {"pattern": keys_patterns, "find_change": [(re.compile(r'[-./\s]'), "")]},
    "UNCNOWN_NUM": {"pattern": uncknow_num_patterns, "mask": " NUMBER ", "lem": False},
}

tokenizer = TextsTokenizer()
syns_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", "synonyms.csv"), sep="\t")

stopwords = []
stopwords_roots = [os.path.join(PROJECT_ROOT_DIR, "data", "greetings.csv"),
                   os.path.join(PROJECT_ROOT_DIR, "data", "stopwords.csv")]

for root in stopwords_roots:
    stopwords_df = pd.read_csv(root, sep="\t")
    stopwords += list(stopwords_df["stopwords"])

tokenizer.add_stopwords(stopwords)

synonyms = [(a, d) for a, d in zip(syns_df["asc"], syns_df["dsc"])]
# tokenizer.add_synonyms(synonyms)
# parser = TextParser(patterns_dict, tokenizer)
# пустой парсер (чтобы привести в соответствие с лемматизированными текстами в Эластике):
parser = TextParser({}, tokenizer)

if __name__ == "__main__":
    txs0 = ["кнд 334455 форме 6 форму ефс-1 формируют салбдо 2023 3. формы формировании отчета 6 формы 26 форме 2 "
           "форме с-09",
           "фсбу 6/01 фсбу 5/2019 фсбу 5-2019 фсбу6/2020 фсбу 25/208 фсбу 6/200 фсбу5/2019",
           "декретные в 1 квартале 2022 или во 2 квартале 2022 в ноябре 2022 5 тыс. рублей 50 коп нам сдали 5 млн. руб.",
           "'ст.149 нк привет статья 170 нк ст.93 нк как можно использовать п.1 пп.3 ст.219 нк ндфл согласно ст. 1000 нк п. 4 пп5 ст. 81 нк или п3 ст.5 нк в соответствии ст 93 нк налоговый кодекс рф п.2 ст.149 нк",
           "'159-фз', '10фз', '14 фз', '255фз', '421-фз', '275 фз', '27-фз', '400-фз', '214 фз', '14-фз', '402-фз', '115-фз'"]
    txs1 = ["какое кбк указывать в уведомлении по земельному налогу", "здравствуйте, подскажите по каким реквизитам  платить земельный налог юридическим лицам за 2022г"]
    txs2 = ["какой кбк указывать в уведомлении по транспортному налогу"]
    print("synonyms:", tokenizer.synonyms)
    txs = ["куда платить транспортный налог юридическим лицам",
           "здравствуйте, подскажите по каким реквизитам  платить земельный налог юридическим лицам за 2022г"]
    for tx in txs:
        import time
        t = time.time()
        d = parser(tx)
        print(time.time() - t)
        print("initial text", tx)
        print("clear_text", d["clear_text"])
        print("lem_clear_text", d["lem_clear_text"])
        print(d)
        print("keys patterns:", keys_patterns)
        print(d["KEYS"])
