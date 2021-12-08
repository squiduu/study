import pandas as pd
from konlpy.tag import Komoran, Kkma, Okt
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
from operator import itemgetter
from tqdm import tqdm


def load_data():

    global dataset
    global stopwords
    global hotel_name

    dataset = pd.read_csv(
        "D:/LikeLion/Code/Project2/Data/spell_check_label.csv", encoding="utf-8"
    )

    with open(
        "D:/LikeLion/Code/Project2/Data/kor_stop.txt", "r", encoding="utf-8"
    ) as f:
        list_file = f.readlines()
    stopwords = [line.rstrip("\n") for line in list_file]

    hotel_name = list(dataset.groupby("hotelName").count().index)


def tokenizer(text):
    # josa = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA', 'XR']
    pos_tag = ["NNG", "NNP", "NNB", "NP", "NR", "VA", "MM", "MAG", "XR"]
    tokens_pos = komoran.pos(text)
    tokens_word = []
    for tag in tokens_pos:
        if tag[1] in pos_tag:
            if tag[0] not in stopwords:
                tokens_word.append(tag[0])
    return re.sub("\.", "", " ".join(tokens_word))


def pre_processing():

    global dataset
    global komoran

    komoran = Komoran(userdic="D:/LikeLion/Code/Project2/Data/userdic.txt")

    # dataset['fixed'] = dataset['fixed'].apply(lambda x: re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","",x))
    dataset["fixed"] = dataset["fixed"].apply(lambda x: tokenizer(x))


def get_term_frequency(document, word_dict=None):
    if word_dict is None:
        word_dict = {}
    words = document.split()

    for w in words:
        word_dict[w] = 1 + (0 if word_dict.get(w) is None else word_dict[w])

    return pd.Series(word_dict, dtype="float64").sort_values(ascending=False)


def get_document_frequency(documents):
    dicts = []
    vocab = set([])
    df = {}

    for d in documents:
        tf = get_term_frequency(d)
        dicts += [tf]
        vocab = vocab | set(tf.keys())

    for v in list(vocab):
        df[v] = 0
        for dict_d in dicts:
            if dict_d.get(v) is not None:
                df[v] += 1

    return pd.Series(df, name="df", dtype="float64").sort_values(ascending=False)


def get_tf(docs):
    vocab = {}
    tfs = []
    for d in docs:
        vocab = get_term_frequency(d, vocab)
        tfs += [get_term_frequency(d)]

    stats = []

    for word, freq in vocab.items():
        tf_v = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tf_v += [tfs[idx][word]]

            else:
                tf_v += [0]

        stats.append((word, freq, *tf_v))

    column_name = ["word", "totalFrequency"]

    for i in range(1, len(docs) + 1):
        column_name.append("document" + str(i))

    return pd.DataFrame(stats, columns=column_name).sort_values(
        "totalFrequency", ascending=False
    )


def get_ntf(matrix):
    max_btf = max(matrix["totalFrequency"])
    total_btf = sum(matrix["totalFrequency"])

    col_names = list(matrix.columns)[2:]

    matrix["ntf1"] = matrix["totalFrequency"].apply(lambda x: x / max_btf)

    matrix_ntf2 = matrix[col_names].copy()

    matrix_ntf2 = matrix_ntf2.apply(lambda x: x / total_btf, axis=1)

    matrix_ntf2["ntf2"] = matrix_ntf2.apply(sum, axis=1)

    matrix["ntf2"] = matrix_ntf2["ntf2"]

    return matrix[["word", "totalFrequency", "ntf1", "ntf2"]]


def get_ntf_idf(ntf, df, document_count):

    ntf = ntf.set_index("word")
    ntf_idf = pd.concat([ntf, df], axis=1)

    # def get_btfidf(scores):
    #     return (np.log(scores['totalFrequency']) + 1.0) * np.log(document_count/scores['df'])

    def get_ntf1idf(scores):
        return (np.log(scores["ntf1"]) + 1.0) * np.log(document_count / scores["df"])

    # def get_ntf2idf(scores):
    #     return (np.log(scores['ntf2']) + 1.0) * np.log(document_count/scores['df'])

    # ntf_idf['btf_idf'] = ntf_idf.apply(get_btfidf, axis=1)
    ntf_idf["ntf1_idf"] = ntf_idf.apply(get_ntf1idf, axis=1)
    # ntf_idf['ntf2_idf'] = ntf_idf.apply(get_ntf2idf, axis=1)

    # btf_rank = ntf_idf['btf_idf'].sort_values(ascending=False)
    ntf1_rank = ntf_idf["ntf1_idf"].sort_values(ascending=False)
    # ntf2_rank = ntf_idf['ntf2_idf'].sort_values(ascending=False)

    # return btf_rank, ntf1_rank, ntf2_rank
    return ntf1_rank


def calculate_save():
    for h_name in tqdm(hotel_name):
        docs = dataset[dataset["hotelName"] == h_name]["fixed"].to_list()

        document_count = len(docs)

        tf_matrix = get_tf(docs)
        df_matrix = get_document_frequency(docs)

        ntf_matrix = get_ntf(tf_matrix)

        # btf_rank , ntf1_rank, ntf2_rank = get_ntf_idf(ntf_matrix, df_matrix, document_count)
        ntf1_rank = get_ntf_idf(ntf_matrix, df_matrix, document_count)

        # btf_rank.to_csv('../../Data/keywords/btf_'+h_name+'.csv', encoding='utf-8-sig')

        ntf1_rank.to_csv(
            "../../Data/keywords/ntf1_" + h_name + ".csv", encoding="utf-8-sig"
        )

        # ntf2_rank.to_csv('../../Data/keywords/ntf2_'+h_name+'.csv', encoding='utf-8-sig')


if __name__ == "__main__":
    load_data()
    pre_processing()
    calculate_save()
