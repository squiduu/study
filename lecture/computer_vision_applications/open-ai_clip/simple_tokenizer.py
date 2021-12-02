import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


# 데코레이터 : 함수의 결과를 복사해서 새 메모리에 넣지 않고, 함수의 결과를 캐시하여 최적화 가능
# This decorator caches the result of the function and optimize the process
@lru_cache()
def default_bpe():
    # 이미 토크나이저 거쳐서 subword로 나누어진 텍스트 파일이 있는 경로 반환
    # returns the directory path where the text file is already divided into subwords through the tokenizer
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.

    bs : ASCII code 0 to 255
    cs : Character corresponding to ASCII code

    Return : {'bs': cs}
    """

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))  # bs : byte string
    cs = bs[:]  # cs : character string
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)  # Convert bytes to string
            n += 1
    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """

    pairs = set()  # Create empty set
    prev_char = word[0]  # First element of word
    for char in word[1:]:
        # iterating elements of word from second to last
        pairs.add((prev_char, char))  # Add tuple (word[i-1], word[i])
        # It can be seen as a bigram applied to words instead of characters.
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)  # Fixing misencoded words.
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)  # strip whitespace
    text = text.strip()
    return text


class SimpleTokenizer(object):
    """
    1) subword를 저장해둔 vocab 파일 -> 디코딩, 분리 -> merge 리스트
    2) 알파벳, 특수문자 모두를 포함하는 ASCII 코드 전체랑 merge랑 합쳐 -> vocab 리스트
    3) NLP 트랜스포머 인풋으로는 vocab이 사용될 예정 (=BPE로 토큰화가 된 subword들이 다 들어있는 리스트)

    1) vocab -> decode, split -> merge list (vocab is the file that subwords are saved)
    2) Merge ASCII code that include alphabet and special chararecters with list name merge and assgin in to a list name vocab 
    3) Vocab will be used to the input of NLP transformer (=List of subwords that are tokenized with BPE)
    """

    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # vocab 파일 압축 해제 -> 읽고 -> 디코딩 -> \n 기준 나누기
        # unzip vocab file then read, decode, split by \n
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        # merges 리스트 원소 하나씩 꺼내서 공백으로 다 나누고 튜플로 된 리스트 만들기
        # split string from merges by whitespace and make a tuple list
        merges = [tuple(merge.split()) for merge in merges]

        # 알파벳, 특수문자 같은 것들을 str 형태로 vocab 에 리스트로 저장
        # assingn alphabet, special character, etc in to a list in form of string type. 
        vocab = list(bytes_to_unicode().values())
        # vocab 토큰화
        # tokenize vocab
        vocab = vocab + [v + '</w>' for v in vocab]

        for merge in merges:
            # merge를 공백없이 합쳐서 토크나이저 vocab 완성
            # Make vocab by joining merge with nothing.
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        # vocab을 활용하기 위한 딕셔너리 만들기
        # make an dictionary to use vocab
        self.encoder = dict(zip(vocab, range(len(vocab))))
        # 인코더 반대로 동작함
        # Works opposite to encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        # merge 파일로 딕셔너리 만들기
        # Make dictionary from merge
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        bigram 기준으로 byte pair encoding을 통한 단어의 토큰화
        Tokenize with byte pair encoding by bigram
        BPE : birthday가 birth 및 -day 로 분리될 수 있는 것처럼, 단어를 의미를 가지는 최소 단위로 분리하여 최대한 처음 보는 토큰이 없도록 만드는 목적
        BPE: Just as birthday can be separated into birthday and -day, the purpose is to separate words into minimal units with meaning so that there are no tokens you see for the first time as much as possible.
        이 함수는 원래 존재하는 BPE 알고리즘과 동일 (참고 - https://wikidocs.net/22592)
        This function is identical to BPE that already exisits.
        """

        if token in self.cache:
            # <startoftext>, <endoftext> 있으면 돌려주기 (얘네 둘 다 원래 트랜스포머에서 문장을 구분하기 위한 용도로 쓰임)
            # Return if <startoftext>, <endoftext> exists 
            return self.cache[token]
        # token을 마지막에 </w> 붙여서 word 튜플로 만들기
        # Add </w> to the end of token then make it to a tuple
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            # pairs에서 bigram 추출 (= bigram 기준으로 bpe 알고리즘을 돌리겠다)
            # Extract bigram from pairs
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word

        # bigram을 기준으로 bpe로 토큰화 한 후, 그 결과를 반환
        # Tokenize to bpe based on bigram and return the result.
        return word

    def encode(self, text):
        """
        토크나이저 인코더 : 공백을 지우고, BPE 과정을 통해 트랜스포머에서 학습 할 수 있는 형태의 토큰을 반환
        Encoder: strip whitespace, return trainable token from bpe
        """

        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))

        return bpe_tokens

    def decode(self, tokens):
        """
        토크나이저 디코더 : 인코더의 키와 밸류를 반대로 돌리고, 토큰화된 subword를 다시 온전한 word 형태로 만들어 반환
        Decoder: Reverse key and value from encoder, returns tokenized subword in a form of complete word.
        """

        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')

        return text
