import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    trans_mod = {}

    #Number of pages in the corpus
    num_files = len(corpus)

    #Number of links from current page
    num_links = len(corpus[page])

    if num_links != 0:
        #Calculating random probability for all pages
        rand_prob = (1 - damping_factor) / num_files
        #Calculating specific page-related probability
        spec_prob = damping_factor / num_links
    else:
        #Calculating random probability for all pages
        rand_prob = (1 - damping_factor) / num_files
        #Calculating specific page-related probability
        spec_prob = 0

    #Iterating over pages
    for file in corpus:
        #Checking if current page has any links
        if len(corpus[page]) == 0:
            trans_mod[file] = 1 / num_files
        else:
            #If not current page, then no need for links
            if file not in corpus[page]:
                #Non-linked page probability is damp
                trans_mod[file] = rand_prob
            else:
                #Probability for linked page will be Specific plus Random probability
                trans_mod[file] = spec_prob + rand_prob
    #Confirming that the sum of probabilities equals 1
    if round(sum(trans_mod.values()), 5) != 1:
        print(f"Error ! Probabilities add up to {sum(trans_mod.values())}")
    return trans_mod


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample_PR = {}
    for page in corpus:
        sample_PR[page] = 0

    #Starting the sample at none
    sample = None

    for iteration in range(n):
        #If in first round the sample is none
        if sample == None:
            #List with all choices
            choices = list(corpus.keys())
            #Choose a sample randomly with random.choice at equal probability
            sample = random.choice(choices)
            sample_PR[sample] += 1
        else:
            #Getting the probability distribution based on current sample
            next_sample_prob = transition_model(corpus, sample, damping_factor)
            #List with all possible choices
            choices = list(next_sample_prob.keys())
            #Getting the weight for choice in choices based 
            weights = [next_sample_prob[key] for key in choices]
            sample = random.choices(choices, weights).pop()
            sample_PR[sample] += 1
    #After finishing sampling, to get the percentages, divide stored values by number of iterations
    sample_PR  = {key: value/n for key, value in sample_PR.items()}
    #Checking if the dictionary values add up to 1
    if round(sum(sample_PR.values()), 5) != 1:
        print(f"Error! Probabilities add up to {sum(trans_mod.values())}")
    else:
        print(f"Sum of sample_pagerank values: {round(sum(sample_PR.values()), 10)}")
    return sample_PR

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #Creating an empty iterate probabilities dict
    iterate_PR = {}
    #Saving the number of pages in a variable
    num_pages = len(corpus)
    #Iterating over all corpus pages assigning 1 dividing by the number of pages
    for page in corpus:
        iterate_PR[page] = 1/ num_pages
    
    changes = 1
    iterations = 1
    while changes >= 0.001:
        #Reseting changes value
        changes = 0
        #Copying the current state to calculate new probabilities without new calculated values
        previous_state = iterate_PR.copy()
        #Iterating over pages
        for page in iterate_PR:
            #Grabing "parent" pages that link to the current page
            parents = [link for link in corpus if page in corpus[link]]
            #Adding the first part of the equation
            firsteq = ((damping_factor - 1)/num_pages)
            #Adding the secound part of the equation by iterating over parents
            secondeq = []

            if len(parents) != 0:
                for parent in parents:
                    #Gathering links starting from parent page
                    num_links = len(corpus[parent])
                    val = previous_state[parent] / num_links
                    secondeq.append(val)

        #Summing values of second list together
        secondeq=sum(secondeq)
        iterate_PR[page] = firsteq + (damping_factor * secondeq)
        #Calculating the the change during iteration
        new_change = abs(iterate_PR[page] - previous_state[page])
        #Updating change value if new_change value if larger
        if changes < new_change:
            changes = new_change
    iterations += 1
    #Normalizing values
    dictsum = sum(iterate_PR.values())
    iterate_PR = {key: value/dictsum for key, value in iterate_PR.items()}
    print(f"\nPage stable after {iterations} iterations.")
    print(f"Sum of iterate_pagerank values: {round(sum(iterate_PR.values()), 10)}")
    return iterate_PR


if __name__ == "__main__":
    main()
