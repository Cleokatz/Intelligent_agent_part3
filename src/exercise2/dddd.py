#!/usr/bin/env python3
from collections import Counter

import numpy as np
import spade
from gensim.corpora import Dictionary
from gensim.matutils import cossim
from array import array
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer

# get your solution from exercise 1!
from src.exercise2.agent_ir import IRAgent # e.g., might rename bidder cause answering added
from src.exercise2.agent_auctioneer import AuctioneerAgent
# new agent type
from src.exercise2.agent_questioner import QuestionerAgent
from src.utils import functions, helpfunctions

# Parameters to create new data
initialized = True  # For downloading in json
preprocessed = False  # For preprocessing either with bigrams (lda=True) or without
gen_lda_model = True  # Create lda_model

def calc_freq(database, corpus_list):
    doc_f = {}
    for title in corpus_list:
        tokens = database[title]
        # For all words in document add title
        for token in np.unique(tokens):
            try:
                doc_f[token].add(title)
            except KeyError:
                doc_f[token] = {title}
    # Calculate the frequency over number of title in which the word appears
    for key in doc_f:
        doc_f[key] = len(doc_f[key])
    return doc_f


def doc_freq(token, doc_f):
    try:
        i = doc_f[token]
    # If word doesn't exist return one
    except KeyError:
        i = 1
    return i


def calc_tf_idf(database, corpus_list):
    tf_idf = {}
    # Calculate the document frequency over the corpus
    doc_f = calc_freq(database, corpus_list)
    for title in corpus_list:
        tokens = database[title]
        # Count number of words in the document
        counter = Counter(tokens)
        # Document length
        words_count = len(tokens)
        # Calculate tf_idf
        for token in np.unique(tokens):
            # Calculate term_frequency
            tf = counter[token] / words_count
            # Get document_frequency for a specific token
            df = doc_freq(token, doc_f)
            # Calculate idf
            idf = np.log(len(corpus_list) / df)
            # Calculate the tf_idf and store them with (title, token) key
            tf_idf[title, token] = tf * idf
    return tf_idf

def valuating(database, query, tf_idf_docs, sell_doc_title):
    value_query = []
    for que in query:
        query_weights = {}
        tokens = database[que]
        # Calculate the value for a document based on a specific query
        for key in tf_idf_docs:
            # Only those are relevant which are in the query
            if key[1] in tokens:
                try:
                    query_weights[key[0]] += tf_idf_docs[key]
                except KeyError:
                    query_weights[key[0]] = tf_idf_docs[key]

        # Sort the values in descending order
        sorted_arr = sorted([query_weights[key] for key in query_weights], reverse=True)
        # Calculate the mean over the tf_idf of document without the highest value (usually the doc from the query)
        mean = np.mean(sorted_arr[1:])
        value_query.append((mean < query_weights[sell_doc_title], query_weights[sell_doc_title] - mean))
    # Calculate value for the selling document based on the query
    value = 0
    # Scale the value based on the individual outputs from the query
    scale_factor = 0
    for val in value_query:
        if val[0]:
            scale_factor += 1
            value += val[1]
    return value * scale_factor/len(query)


async def main():
    if not initialized:
        helpfunctions.download_wiki()
    if not preprocessed:
        helpfunctions.preprocessing_all(lda_preprocessing=True)
    if not gen_lda_model:
        lda_model = None  # gensim.models.LdaMulticore.load("./model/lda_model_corpus")  # None
        helpfunctions.make_lda(lda_model)
        return

    dct, database = helpfunctions.create_dictionaries()

    return
    corpus_list = functions.read_file("./exercise2/data/corpus.txt")
    corpus_list = corpus_list.split("\n")[1:]
    tokenized = functions.read_json_file("./utils/cache/wiki_tokenized.json")
    tf_idf = calc_tf_idf(tokenized, corpus_list)
    #print(tf_idf[("Aston Martin", "abingdon")])


    corpus = [dct.doc2bow(tokenized[key]) for key in corpus_list]  # convert corpus to BoW format
    model = TfidfModel(corpus, id2word=dct, normalize=False, smartirs="nfc")  # fit model
    query = model[[dct.doc2bow(tokenized['Aston Martin'])]]

    # print(query[0])
    sell_list = functions.read_file("./exercise1/data/sell.txt")
    sell_list = sell_list.split("\n")[1:]
    for x in sell_list:
        sell_doc = model[[dct.doc2bow(tokenized[x])]]
        score = cossim(query[0], sell_doc[0])
        ls = corpus_list
        ls.append(x)
        tf_idf = calc_tf_idf(tokenized, ls)
        value = valuating(tokenized, ['Aston Martin'], tf_idf, x)
        if score > 0.015 or value > 0:#0.005:
            if score < 0.015:
                score = 0.0
            print(x, score, value)

    sell_doc2 = model[[dct.doc2bow(tokenized['Banana'])]]
    print(cossim(query[0], sell_doc2[0]))
    # dct = Dictionary([tokenized['Aston Martin']])
    # print(dct.doc2idx(["abingdon"]))
    vector = model[[dct.doc2bow(tokenized['Aston Martin'])]]
    # print(vector[0])
    # TODO: Testen: Wass passiert wenn ich cossim auf gensim mache, kann ich daraus kaufdokumente ableiten?
    return
    # IR agents
    ir1 = IRAgent("bidder1@localhost", "bidder1")
    ir1.set("corpus_file", "corpus")
    await ir1.start()

    ir2 = IRAgent("bidder2@localhost", "bidder2")
    ir2.set("corpus_file", "corpus")
    await ir2.start()

    # Auctioneer
    auctioneer = AuctioneerAgent("auctioneer@localhost", "auctioneer")
    auctioneer.set("bidders_list", ['bidder1@localhost', 'bidder2@localhost'])
    auctioneer.set("documents_file", "sell")
    await auctioneer.start()

    # Questioners
    questioner1 = QuestionerAgent("questioner1@localhost", "questioner1")
    questioner1.set("query_target", "bidder1@localhost")
    questioner1.set("queries_file", "queries_i")
    await questioner1.start()

    questioner2 = QuestionerAgent("questioner2@localhost", "questioner2")
    questioner2.set("query_target", "bidder2@localhost")
    questioner2.set("queries_file", "queries_ii")
    await questioner2.start()


if __name__ == "__main__":
    spade.run(main())
