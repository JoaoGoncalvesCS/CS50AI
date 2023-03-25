import nltk
import sys
import os
import pandas as pd
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    #Initiating dict
    f_con = dict()

    for j, _, k in os.walk(directory):
        for i in k:
            f = open(os.path.join(j, i), encoding="utf8")
            f_con[i] = f.read()

    return f_con


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punc =string.punctuation
    stop_words = nltk.corpus.stopwords.words("english")
    w = nltk.word_tokenize(document.lower())
    w = [i for i in w if i not in punc and i not in stop_words]
    return w


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    #Number of dicts or files
    idfs = dict()

    t_n_d = len(documents)
    words = set(word for sublist in documents.values() for word in sublist)

    for word in words:
        num = 0
        for document in documents.values():
            if word in document:
                num += 1
        idf = math.log(t_n_d / num)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    #Initiating tfidf-dict
    file_scores = dict()
    for i, j in files.items():
        total_tf_idf = 0
        for word in query:
            total_tf_idf += j.count(word) * idfs[word]
        file_scores[i] = total_tf_idf
    ranked_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_files = [x[0] for x in ranked_files]

    return ranked_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    #Initiating a tfidf dict
    s_score = dict()
    for i, j in sentences.items():
        in_query = query.intersection(j)
        idf = 0
        for word in in_query:
            idf += idfs[word]
        num = sum(map(lambda x: x in in_query, j))
        s_score[i] = {"idf": idf, "qtd": num / len(j)}
    r_sen = sorted(s_score.items(), key=lambda x: (x[1]["idf"], x[1]["qtd"]), reverse=True)
    r_sens = [x[0] for x in r_sen]

    return r_sens[:n]


if __name__ == "__main__":
    main()