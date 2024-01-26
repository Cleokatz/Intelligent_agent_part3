import os

import gensim
#import pyLDAvis
#import pyLDAvis.gensim
from gensim.corpora import dictionary
from gensim.matutils import cossim
from gensim.models import TfidfModel
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
from nltk import RegexpTokenizer, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

import src.utils.wikipedia as wiki
from src.utils import functions

# Get hellinger distances between the corpus and the query
def get_distances(corpus_titles, query_title, lda_model, database_all):
    # Query to bow format
    database_arr = []
    title_corpus_distances = []
    id2word = gensim.corpora.Dictionary([database_all[query_title]])
    corpus = [id2word.doc2bow(t) for t in [database_all[query_title]]]
    # Get vector of the query
    query_vec = lda_model.get_document_topics(corpus)
    # Optional print
    # print(lda_model.print_topics())
    # print(query_vec[0])
    for title in corpus_titles:
        database_arr.append(database_all[title])
        # Each title to bow format, and store the distance between the document and the query for ranking
        id2word = gensim.corpora.Dictionary([database_all[title]])
        corpus = [id2word.doc2bow(t) for t in [database_all[title]]]
        title_corpus_distances.append((title, gensim.matutils.hellinger(lda_model.get_document_topics(corpus)[0], query_vec[0])))
    # Complete  corpus to bow format
    id2word = gensim.corpora.Dictionary(database_arr)
    corpus = [id2word.doc2bow(t) for t in database_arr]

    corpus_vec = lda_model.get_document_topics(corpus)
    # Return the general distance to the corpus and the distances to the documents
    return gensim.matutils.hellinger(corpus_vec[0], query_vec[0]), title_corpus_distances


# Return documents based on specific distances
def get_discrepancy(distances, corpus_title, database_all, query_title, dct):
    distances[1].sort(key=lambda dist: dist[1])
    # Optional print
    #print(query_title)
    #print(distances)
    if distances[0] <= 0.35:
        documents = []
        for que in query_title:
            if que in corpus_title:
                documents.append(que)
        for tu in distances[1]:
            documents.append(tu[0])
        return documents[:min(10, len(corpus_title))]
    elif distances[0] <= 0.5:
        docs = []
        if query_title in corpus_title:
            docs.append(query_title)
        corpus_title.extend(query_title)
        documents = []
        for key in distances[1]:
            score = calculate_score(dct, database_all, corpus_title, query_title, key[0])
            value = 10 * score ** (1 / 100)
            documents.append((key[0], key[1]*value))
        # Optional print
        #print(documents)
        documents.sort(key=lambda dist: dist[1])
        for d in documents:
            docs.append(d[0])
        return docs[:min(10, len(corpus_title))]
    else:
        return "Incompatible, please ask another agent"


# Calculate value for agents
def calculate_score(dct: dictionary.Dictionary, database: dict, corpus_list: list[str], query: list[str], buy_doc: str) -> float:
    # Valuating the item based on the tf_idf score
    corpus = [dct.doc2bow(database[key]) for key in corpus_list]  # convert corpus to BoW format
    model = TfidfModel(corpus, id2word=dct, normalize=False, smartirs="nfc")  # fit model
    query_corpus = [dct.doc2bow(database[key]) for key in query]
    query = model[query_corpus]
    sell_doc = model[[dct.doc2bow(database[buy_doc])]]
    return cossim(query[0], sell_doc[0])


# Create new lda model based on corpus
def create_new_model(dct: dictionary.Dictionary, database: dict, corpus_list: list[str]):
    corpus = [dct.doc2bow(database[key]) for key in corpus_list]
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dct,
                                           num_topics=20,
                                           # random_state=100,
                                           workers=3,
                                           iterations=1000,
                                           minimum_phi_value=0.001,
                                           minimum_probability=0.001,
                                           gamma_threshold=0.001,
                                           passes=2,
                                           per_word_topics=True,
                                           alpha=0.01,
                                           eta=0.1)
    return lda_model


# Make the lda models and store it, if there is not already a given model. Compute coherence and perplexity of the model
def make_lda(lda_model_akt: bool = None):
    # Get dct and database
    dct, database = create_dictionaries()
    database_arr = [database[key] for key in database]
    corpus = [dct.doc2bow(database[key]) for key in database]
    if lda_model_akt is None:
        # Make lda model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=dct,
                                               num_topics=20,
                                               #random_state=100,
                                               workers=3,
                                               iterations=1000,
                                               minimum_phi_value=0.001,
                                               minimum_probability=0.001,
                                               gamma_threshold=0.001,
                                               passes=2,
                                               per_word_topics=True,
                                               alpha=0.01,
                                               eta=0.1)

        lda_model.save("./model/lda_model_corpus")
        lda_model_akt = lda_model

    # Compute Perplexity: measure of how good the model is. lower the better.
    print('\nPerplexity: ', lda_model_akt.log_perplexity(corpus))
    # Compute Coherence Score: measure of how good the model is. higher the better.
    c_v_mod = gensim.models.CoherenceModel(model=lda_model_akt, texts=database_arr, dictionary=dct, coherence='c_v')
    coherence_lda = c_v_mod.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    # Uncomment for visualizing the lda model
    vis = pyLDAvis.gensim.prepare(lda_model_akt, corpus, dct)
    pyLDAvis.save_html(vis, 'lda.html')


# Get value for a specific score
def get_value(score: float, opponent_score: float, strategy, distances, buy_doc) -> int:
    if score <= 0.015:
        return 0
    elif strategy == "standard":
        x = 100 * score ** (1/100)
        value = x if x < 96.1 else max(96.1 + score*2, 100)
        return round(value)  # round(score**(1/100) * 5, 4)
    elif strategy == "money_optimized":
        if opponent_score <= 0.008:
            return 1
        else:
            x = 10 * opponent_score ** (1 / 100)
            return round(x) + 2
    elif strategy == "reward_orient":
        distances[1].sort(key=lambda dist: dist[1])
        dist = distances[1]
        val = 50
        #print(distances)
        #print(buy_doc)
        for el in dist[:20]:
            #print(el[0])
            #print(buy_doc == el[0])
            if buy_doc == el[0]:
                return val
            val -= 5
        if val == 0:
            val = 1
        return max(round(val*score), 100)


# Create a gensim dictionary to provide a gensim bow for the models, and a database dictionary
def create_dictionaries() -> tuple[dictionary.Dictionary, dict]:
    path = "./utils/cache/"
    path_list = os.listdir(path)
    dct = dictionary.Dictionary()
    database = {}
    for li in path_list:
        # Only get the tokenized documents
        if "tokenized" in li:
            tokens = functions.read_json_file(path + li)
            database.update(tokens)
            for key in tokens:
                dct.add_documents([tokens[key]])
    return dct, database


# Preprocess text into tokens after specific procedure
def preprocessing(text: str, preprocessing_lda: bool) -> list[str]:
    if preprocessing_lda:
        # Make text to tokens
        data_words = list(gensim_preprocess(text))
        # Get bigram model
        bigram = gensim.models.Phrases(sentences=data_words, min_count=20,
                                       connector_words=ENGLISH_CONNECTOR_WORDS)
        # Make it static
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # Filter out stopwords
        filtered_token = [token for token in gensim.utils.simple_preprocess(str(text)) if
                          token not in stopwords.words('english')]
        # Create additionally bigrams
        bigram_token = bigram_mod[filtered_token]
        other_token = []
        for token in bigram_token:
            if token not in filtered_token:
                other_token.append(token)
        filtered_token.extend(other_token)
        # Lemmatize tokens
        lemmatize = WordNetLemmatizer()
        processed_token = [lemmatize.lemmatize(token) for token in filtered_token]
    else:
        # Make text to tokens and remove punctuation with regex-pattern: w for word-char, + for many w
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        # Convert to lowercase and filter out all tokens that are non-alphabetic
        cleared_token = [token.lower() for token in tokens if token.isalpha()]
        # Filter out stop words
        filtered_token = [token for token in cleared_token if token not in stopwords.words('english')]
        # Stem tokens
        stemmer = PorterStemmer()
        processed_token = [stemmer.stem(token) for token in filtered_token]
    return processed_token


# Yield the preprocess for gensim
def gensim_preprocess(text: str) -> list[str]:
    yield gensim.utils.simple_preprocess(str(text), deacc=True, max_len=100)


# Processes all document-corpora to tokenized corpora and store the tokenized documents in json
def preprocessing_all(lda_preprocessing: bool = False):
    path = "./utils/cache/"
    path_list = os.listdir(path)
    for li in path_list:
        # Load database
        database = functions.read_json_file(path + li)
        database_processed = {}
        if "tokenize" not in li:
            for key in database:
                database_processed[key] = preprocessing(database[key], lda_preprocessing)
            functions.write_json_file(path + li[:-5] + "_tokenized.json", database_processed)


# Download necessary wiki-articles and store in cache
def download_wiki():
    path = "./exercise3/data/"
    path_list = os.listdir(path)
    for li in path_list:
        # Create directory with class wikipedia
        wikit = wiki.Wikipedia(li[:-4])
        # Download corpus list
        content = functions.read_file(path + li)
        content = content.split("\n")
        # [1:] because first object is empty
        for c in content[1:]:
            wikit.get(c)
