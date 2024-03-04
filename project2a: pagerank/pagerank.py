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
    linked to `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    page_links = corpus[page]
    num_links = len(page_links)

    if num_links == 0:
        return {page: 1 / num_pages for page in corpus}

    transition_prob = {page: damping_factor / num_links for page in page_links}

    for page in corpus:
        transition_prob[page] = transition_prob.get(page, 0) + (1 - damping_factor) / num_pages

    return transition_prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    ranks = {page: 0 for page in corpus}

    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        ranks[current_page] += 1
        transition_prob = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(list(transition_prob.keys()), weights=list(transition_prob.values()), k=1)[0]
        current_page = next_page

    total_ranks = sum(ranks.values())
    for page in ranks:
        ranks[page] /= total_ranks

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    ranks = {page: 1 / num_pages for page in corpus}
    new_ranks = {page: 0 for page in corpus}

    while True:
        for page in corpus:
            rank_sum = sum(ranks[link] / len(corpus[link]) for link in corpus if page in corpus[link])
            new_ranks[page] = (1 - damping_factor) / num_pages + damping_factor * rank_sum

        is_converged = all(abs(new_ranks[page] - ranks[page]) < 0.001 for page in corpus)
        if is_converged:
            break

        ranks = new_ranks.copy()

    return ranks


if __name__ == "__main__":
    main()

