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
    topics = {}

    #Definning path to directory
    path_to_dir = os.path.join(".", f"{directory}")

    #Iterating over files in directory
    for file in os.listdir(path_to_dir):
        #Getting file path
        path_to_file = os.path.join(path_to_dir, file)
        #Reading file into string
        with open(path_to_file, "r", encoding="utf8") as f:
            string = f.read()

        #Saving text for each file in topics-dict
        topics[file[:-4]] = string

    #Returning dict
    return topics


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = [word.lower() for word in nltk.word_tokenize(document)]

    #Initiating filtered-list
    filtered = []

    #Defining stopwords and puntuation
    stopwords = nltk.corpus.stopwords.words("english")
    punct = [punct for punct in string.punctuation]

    #Iterating over tokens and filtering out stopwords and punctuation
    for word in tokens:
        if word in stopwords:
            continue
        elif word in punct:
            continue
        else:
            filtered.append(word)
    #Returning filtered list
    return (filtered)


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    #Number of dicts or files
    numDict = len(documents)

    #Initiating presence dict
    presence = {}

    #Initiating idfs dict to save the calculated idfs
    idfs = {}

    #Iterating over documents and unique words within each document
    for doc in documents:
        for word in set(documents[doc]):
            if word in presence.keys():
                presence[word] += 1
            else:
                presence[word] = 1

    for word in presence:
        idf = 1 + (math.log(numDict/(presence[word])))
        idfs[word] = idf
    #Returning idf-dict
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    #Initiating tfidf-dict
    tfidfs = {}

    #Iterating over files and words in query and calculating tfidf per file-queryword pair
    for file in files:
        tfidfs[file] = 0
        tokens_in_file = len(files[file])
        for word in query:
            if word in files[file]:
                frequency = files[file].count(word)+1 #From smothing
            else:
                frequency = 1
            tf = frequency/tokens_in_file #Normalizing frequency to account for different length of texts
            if word in idfs.keys():
                idf = idfs[word]
            else:
                idf = 1
            tfidfs[file] += idf * tf #Summing tfidfs from different words togheter per file

    #Creating a list with sorted files
    sorted_list = sorted(tfidfs, key=tfidfs.get, reverse=True)

    #Creating a list with n top files
    topFiles = sorted_list[:n]

    #Returning the list of topFiles
    return topFiles


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    #Initiating a tfidf dict
    sentence_stats = {}

    #Iterating over files and words in the query and calculating the tfidf per file-query word pair
    for sentence in sentences:
        sentence_stats[sentence] = {}
        sentence_stats[sentence]["idf"] = 0
        sentence_stats[sentence]["word_count"] = 0
        senlength = len(sentences[sentence])
        for word in query:
            if word in sentences[sentence]:
                sentence_stats[sentence]["idf"] += idfs[word]
                sentence_stats[sentence]["word_count"] += 1
        sentence_stats[sentence]["QTD"] = float(sentence_stats[sentence]["word_count"] / senlength)

    #Creating a list with sorted sentences
    sorted_list = sorted(sentence_stats.keys(), key=lambda sentence: (sentence_stats[sentence]["idf"], sentence_stats[sentence]["QTD"]), reverse=True)

    #Creating a list with n top sentences
    topSens = sorted_list[:n]

    #Returning topSens
    return topSens


if __name__ == "__main__":
    main()