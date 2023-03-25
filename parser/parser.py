import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
 P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S conj S | VP NP | S P S | S NP | S P NP
NP -> N | Det AA N | Det N | NP Adv V | AA N | Det N AA | P NP
AA -> Adj | Adj AA | Adv
VP -> V | V P NP | Adv V | V P | V AA
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    print(f"Original sentence: {sentence}")
    tokens = nltk.word_tokenize(sentence)
    #Making all word-tokens lowercase
    tokens = [word.lower() for word in tokens]
    print(f"All tokens: {tokens}")
    #Alphabetic filter
    filteralpha = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    filtered = []
    #Iterating over words adding only words with at least 1 letter to filtered list
    for word in tokens:
        for letter in word:
            if letter in filteralpha:
                filtered.append(word)
                break
            else:
                continue
    print(f"Filtered tokens: {filtered}")
    return filtered


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    #Creating a list to save NPs
    NPS = []
    #Iterating over all subtrees
    for subtree in tree.subtrees():
        if (subtree.label() == "NP"):
            NPS.append(subtree)
    #Returning NP
    return NPS

if __name__ == "__main__":
    main()
