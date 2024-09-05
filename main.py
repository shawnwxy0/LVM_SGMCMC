#!/Users/sam/anaconda/bin/python
from __future__ import division

import argparse
import itertools
import numpy as np
from collections import Counter
from itertools import islice

import processwiki as pw
import samplers



def main():
    parser = argparse.ArgumentParser(description="Arguments for running SGRLD for LDA")

    # Optional arguments with default values
    parser.add_argument('--alpha', type=float, default=0.01, help='Dirichlet prior for document-topic distribution')
    parser.add_argument('--beta', type=float, default=0.0001, help='Dirichlet prior for topic-word distribution')
    parser.add_argument('--K', type=int, default=50, help='Number of topics')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Step size')
    parser.add_argument('--tau0', type=float, default=1000, help='Delay parameter for learning rate')
    parser.add_argument('--kappa', type=float, default=0.55, help='Exponential decay for learning rate')
    parser.add_argument('--samples_per_update', type=int, default=10, help='Samples per update')
    parser.add_argument('--num_updates', type=int, default=1000, help='Number of updates')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output')

    # Parse arguments
    args = parser.parse_args()

    # Step size parameters
    step_size_params = (args.epsilon, args.tau0, args.kappa)

    # Number of documents (approx)
    D = 3.3e6  # Example: Approximate number of documents in Wikipedia
    # Clean and process the document according to the vocabulary
    def clean_doc(doc, vocab):
        return [word for word in doc.split() if word in vocab]

    # Simulate Documents
    def simulate_docs(vocab, num_docs=5, doc_length=20):
        np.random.seed(42)  # For reproducibility
        vocab_size = len(vocab)
        docs = []

        for i in range(num_docs):
            doc_words = np.random.choice(list(vocab.keys()), doc_length)
            doc_name = f"doc_{i + 1}"
            docs.append((" ".join(doc_words), doc_name))

        return docs

    # Parse and split the documents into training/test sets
    def parse_docs(name_docs, vocab):
        for doc, name in name_docs:
            words = clean_doc(doc, vocab)
            # Hold back every 10th word for the test set
            train_words = [vocab[w] for (i, w) in enumerate(words) if i % 10 != 0]
            test_words = [vocab[w] for (i, w) in enumerate(words) if i % 10 == 0]
            train_cntr = Counter(train_words)
            test_cntr = Counter(test_words)
            yield (name, train_cntr, test_cntr)

    # Function to batch the documents for online learning
    def take_every(n, iterable):
        i = iter(iterable)
        piece = list(islice(i, n))
        while piece:
            yield piece
            piece = list(islice(i, n))

    # Create Simulated Data for SGRLD
    def simulate_sgrld_data(num_docs=10000, doc_length=1000, batch_size=20):
        # Step 1: Create Vocabulary
        vocab = pw.create_vocab('wiki.vocab')
        # Step 2: Simulate Documents
        docs = simulate_docs(vocab, num_docs, doc_length)

        # Step 3: Parse Documents
        parsed_docs = list(parse_docs(docs, vocab))

        # Step 4: Batch the Documents
        batched_docs = list(take_every(batch_size, parsed_docs))

        return vocab, batched_docs

    # Simulate data with 5 documents
    vocab, batched_docs = simulate_sgrld_data()
    W = len(vocab)
    # Output to check
    # print("Vocabulary:", vocab)
    print("\nBatched Docs (Training and Test Word Counts):")
    flattened_docs = [doc for batch in batched_docs for doc in batch]
    # Unpack the flattened docs into test_names, holdout_train_cts, and holdout_test_cts
    test_names, holdout_train_cts, holdout_test_cts = zip(*flattened_docs)

    # Convert holdout_train_cts and holdout_test_cts to lists if needed
    holdout_train_cts = list(holdout_train_cts)
    holdout_train_cts = list(holdout_train_cts)
    batched_cts = take_every(args.batch_size, batched_docs)
    print ("Running LD Sampler, epsilon = %g, samples per update = %d" % (args.epsilon,
            args.samples_per_update))

    theta0 = np.random.gamma(1,1,(args.K,W))
    ld_func = samplers.LDSampler(D, args.K, W, args.alpha, args.beta, theta0,
            step_size_params, args.output_dir).run_online
    ld_args = (args.num_updates, args.samples_per_update, batched_cts,
            holdout_train_cts, holdout_test_cts)
    ld_func(*ld_args)


if __name__ == '__main__':
    main()
