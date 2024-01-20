import re
import collections
from heapq import heappush, heappop
from typing import Optional, Union, List, Tuple, Dict

class TextPreprocessor:
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def preprocess(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        return text

class BaseTokenizer:
    def __init__(self, corpus: Optional[Union[List[str], str]] = None, lowercase: bool = True, remove_punctuation: bool = True):
        self.vocab = {}
        self.preprocessor = TextPreprocessor(lowercase, remove_punctuation)
        if corpus:
            self.add_corpus(corpus)

    def add_corpus(self, corpus: Union[List[str], str]) -> None:
        if isinstance(corpus, str):
            corpus = [corpus]
        for text in corpus:
            self.tokenize_corpus(text)

    def tokenize_corpus(self, text: str) -> None:
        processed_text = self.preprocessor.preprocess(text)
        tokens = processed_text.split()
        for token in tokens:
            self.get_or_add_token_id(token)

    def tokenize_sentence(self, sentence: str) -> List[int]:
        return [self.get_or_add_token_id(token) for token in sentence.split()]

    def get_or_add_token_id(self, token: str) -> int:
        if token not in self.vocab:
            token_id = len(self.vocab) + 1
            self.vocab[token] = token_id
        return self.vocab[token]

    def tokenize(self, text: Union[List[str], str], padding: bool = False, max_length: Optional[int] = None) -> Union[List[List[int]], List[int]]:
        if isinstance(text, str):
            text = [text]
        tokenized_text = [self.tokenize_sentence(sentence) for sentence in text]
        if padding:
            max_len = max(len(tokens) for tokens in tokenized_text)
            tokenized_text = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized_text]
        if max_length is not None:
            tokenized_text = [tokens[:max_length] for tokens in tokenized_text]
        return tokenized_text

    def __call__(self, text: Union[List[str], str], padding: bool = False, max_length: Optional[int] = None) -> Union[List[List[int]], List[int]]:
        return self.tokenize(text, padding, max_length)

class WordTokenizer(BaseTokenizer):
    def __init__(self, corpus: Optional[Union[List[str], str]] = None, lowercase: bool = True, remove_punctuation: bool = True):
        super().__init__(corpus, lowercase, remove_punctuation)


    def train(self, n_iter: int) -> None:
        # WordTokenizer에는 특별한 training이 필요 없으므로 이 메소드는 아무 작업도 수행하지 않습니다.
        pass

class BPETokenizer(BaseTokenizer):
    def __init__(self, corpus: Optional[Union[List[str], str]] = None, vocab_size: int = 10000, lowercase: bool = True, remove_punctuation: bool = True):
        super().__init__(corpus, lowercase, remove_punctuation)
        self.vocab_size = vocab_size

    def get_stats(self) -> Dict:
        pairs = collections.defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str]) -> None:
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in list(self.vocab.keys()):
            w_out = p.sub(''.join(pair), word)
            if w_out != word:
                self.vocab[w_out] = self.vocab.pop(word)

    def train(self, n_iter: int) -> None:
        for _ in range(n_iter):
            pairs = self.get_stats()
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break

            self.merge_vocab(best)

            # 병합된 pair에 대한 빈도 업데이트
            self.update_frequencies(best)

    def update_frequencies(self, merged_pair: Tuple[str, str]) -> None:
        merged_word = ''.join(merged_pair)
        words_to_remove = set()
        words_to_add = collections.defaultdict(int)

        for word, freq in self.vocab.items():
            if merged_pair[0] in word and merged_pair[1] in word:
                new_word = word.replace(' '.join(merged_pair), merged_word)
                words_to_remove.add(word)
                words_to_add[new_word] += freq

        for word in words_to_remove:
            del self.vocab[word]

        for word, freq in words_to_add.items():
            self.vocab[word] += freq
