from typing import List

import pkuseg


class StopwordUtil:
    def __init__(self, stopwords_path: str):
        self.seg = pkuseg.pkuseg()
        with open(stopwords_path, encoding='utf8') as f:
            self.stopwords = set(k.strip() for k in f if k.strip())

    def chinese_tokenizer(self, text: str) -> List[str]:
        tokens = self.seg.cut(text)
        return [token for token in tokens if token not in self.stopwords]
