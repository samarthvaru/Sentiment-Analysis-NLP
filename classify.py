import math, re, sys
import numpy as np

def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


def process(x):
    t=[]
    for i in x:
        if i=="positive":
            t.append(1)
        if i=="negative":
            t.append(0)
        if i=="neutral":
            t.append(2)
    return t
            
def baseline(pos_words,neg_words,test_texts):
        class_for_test=[]
        for i in test_texts:
            count_pos=0
            count_neg=0
            count_neu=0
            tokens=tokenize(i)
            for t in tokens:
                if t in pos_words:
                    count_pos+=1
                elif t in neg_words:
                    count_neg+=1
                else:
                    count_neu+=1
            if count_pos>count_neg:
                class_for_test.append('positive')
            elif count_pos<count_neg:
                class_for_test.append('negative')
            elif count_pos==count_neg:
                class_for_test.append('neutral')

        return class_for_test

 
class MNNaiveBayes:

    def __init__(self, k=1):
        self.k = k
        self.cat0_count = 0
        self.cat1_count = 0
        self.cat2_count = 0
        self.total_count = 0
        self.cat_0_prior = 0
        self.cat_1_prior = 0
        self.cat_2_prior = 0
        self.cat_0_prior, self.cat_1_prior, self.cat_2_prior
        self.word_probs = []
        self.vocab = []

    def tokenize(self,s):
        tokens=s.lower().split()
        trimmed_tokens = []
        for t in tokens:
            if re.search('\w', t):
                # t contains at least 1 alphanumeric character
                t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
                t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
            trimmed_tokens.append(t)
        return trimmed_tokens
  
    def count_words(self, X, y):
        counts = {}
        # need to figure our this loop, want to iterate over both of them, I see why it was paired before
        for document, category in zip(X, y):
            for token in self.tokenize(document):
              # Initialize a dict entry with 0 counts
              if token not in counts:
                counts[token] = [0,0,0]
              # Now that it exists, add to the category count for that word
              counts[token][category] += 1
        return counts

    def prior_prob(self, counts):
        
        # Iterate through counts dict and add up each word count by category
        cat0_word_count = cat1_word_count = cat2_word_count= 0
        for word, (cat0_count, cat1_count,cat2_count) in counts.items():
            cat0_word_count += cat0_count
            cat1_word_count += cat1_count
            cat2_word_count += cat2_count

        # save attributes to the class
        self.cat0_count = cat0_word_count
        self.cat1_count = cat1_word_count
        self.cat2_count= cat2_word_count
        self.total_count = self.cat0_count + self.cat1_count + self.cat2_count
        
        # Get the prior prob by dividing words in each cat by total words
        cat_0_prior = cat0_word_count / self.total_count
        cat_1_prior = cat1_word_count / self.total_count
        cat_2_prior = cat2_word_count / self.total_count
        return cat_0_prior, cat_1_prior, cat_2_prior

    def check_probs(self):
        assert self.cat_0_prior+self.cat_1_prior+self.cat_2_prior==1

    def word_probabilities(self, counts):
        """turn the word_counts into a list of triplets
        word, p(w | cat0), and p(w | cat1)"""
        # Here we apply the smoothing term, self.k, so that words that aren't in
        # the category don't get calculated as 0
        self.vocab = [word for word, (cat0, cat1,cat2) in counts.items()]
        l=len(self.vocab)
        return [(word,
        (cat0 + self.k) / (self.cat0_count + l),
        (cat1 + self.k) / (self.cat1_count + l),
        (cat2 + self.k) / (self.cat2_count + l))
        for word, (cat0, cat1, cat2) in counts.items()]

    def fit(self, X, y):
        # Take all these functions and establish probabilities of input
        counts = self.count_words(X, y)
        self.cat_0_prior, self.cat_1_prior, self.cat_2_prior = self.prior_prob(counts)
        self.word_probs = self.word_probabilities(counts)

    def predict(self, test_corpus):
        # Split the text into tokens,
        # For each category: calculate the probability of each word in that cat
        # find the product of all of them and the prior prob of that cat
        y_pred = []
        for document in test_corpus:
          # Every document get their own prediction probability
          log_prob_cat0 = log_prob_cat1 =log_prob_cat2 = 0.0
          tokens = self.tokenize(document)
            # Iterate through the training vocabulary and add any log probs that match
            # if no match don't do anything. We just need a score for each category/doc
          for word, prob_cat0, prob_cat1, prob_cat2 in self.word_probs:
            if word in tokens:
              # Because of 'overflow' best to add the log probs together and exp
              log_prob_cat0 += math.log(prob_cat0)
              log_prob_cat1 += math.log(prob_cat1)
              log_prob_cat2 += math.log(prob_cat2)
            # get each of the category predictions including the prior
          cat_0_pred = self.cat_0_prior * math.exp(log_prob_cat0)
          cat_1_pred = self.cat_1_prior * math.exp(log_prob_cat1)
          cat_2_pred = self.cat_2_prior * math.exp(log_prob_cat2)
          if cat_0_pred >= cat_1_pred and cat_0_pred >= cat_2_pred:
              y_pred.append(0)
          elif cat_1_pred >= cat_0_pred and cat_1_pred >= cat_2_pred:
              y_pred.append(1)

          elif cat_2_pred >= cat_1_pred and cat_2_pred >= cat_0_pred:
              y_pred.append(2)
        return y_pred



class MNNaiveBayesBinary:

    def __init__(self, k=1):
        self.k = k
        self.cat0_count = 0
        self.cat1_count = 0
        self.cat2_count = 0
        self.total_count = 0
        self.cat_0_prior = 0
        self.cat_1_prior = 0
        self.cat_2_prior = 0
        self.cat_0_prior, self.cat_1_prior, self.cat_2_prior
        self.word_probs = []
        self.vocab = []

    def tokenize(self,s):
        tokens=s.lower().split()
        trimmed_tokens = []
        for t in tokens:
            if re.search('\w', t):
                # t contains at least 1 alphanumeric character
                t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
                t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
            trimmed_tokens.append(t)
        return trimmed_tokens
  
    def count_words(self, X, y):
        counts = {}
        # need to figure our this loop, want to iterate over both of them, I see why it was paired before
        for document, category in zip(X, y):
            vis=[]
            for token in self.tokenize(document):
              if token in vis:
                  continue
              # Initialize a dict entry with 0 counts
              if token not in counts:
                counts[token] = [0,0,0]
              # Now that it exists, add to the category count for that word
              counts[token][category] += 1
              vis.append(token)
        return counts

    def prior_prob(self, counts):
        
        # Iterate through counts dict and add up each word count by category
        cat0_word_count = cat1_word_count = cat2_word_count= 0
        for word, (cat0_count, cat1_count,cat2_count) in counts.items():
            cat0_word_count += cat0_count
            cat1_word_count += cat1_count
            cat2_word_count += cat2_count

        # save attributes to the class
        self.cat0_count = cat0_word_count
        self.cat1_count = cat1_word_count
        self.cat2_count= cat2_word_count
        self.total_count = self.cat0_count + self.cat1_count + self.cat2_count
        
        # Get the prior prob by dividing words in each cat by total words
        cat_0_prior = cat0_word_count / self.total_count
        cat_1_prior = cat1_word_count / self.total_count
        cat_2_prior = cat2_word_count / self.total_count
        return cat_0_prior, cat_1_prior, cat_2_prior

    def check_probs(self):
        assert self.cat_0_prior+self.cat_1_prior+self.cat_2_prior==1

    def word_probabilities(self, counts):
        """turn the word_counts into a list of triplets
        word, p(w | cat0), and p(w | cat1)"""
        # Here we apply the smoothing term, self.k, so that words that aren't in
        # the category don't get calculated as 0
        self.vocab = [word for word, (cat0, cat1,cat2) in counts.items()]
        l=len(self.vocab)
        return [(word,
        (cat0 + self.k) / (self.cat0_count + l),
        (cat1 + self.k) / (self.cat1_count + l),
        (cat2 + self.k) / (self.cat2_count + l))
        for word, (cat0, cat1, cat2) in counts.items()]

    def fit(self, X, y):
        # Take all these functions and establish probabilities of input
        counts = self.count_words(X, y)
        self.cat_0_prior, self.cat_1_prior, self.cat_2_prior = self.prior_prob(counts)
        self.word_probs = self.word_probabilities(counts)

    def predict(self, test_corpus):
        # Split the text into tokens,
        # For each category: calculate the probability of each word in that cat
        # find the product of all of them and the prior prob of that cat
        y_pred = []
        for document in test_corpus:
          # Every document get their own prediction probability
          log_prob_cat0 = log_prob_cat1 =log_prob_cat2 = 0.0
          tokens = self.tokenize(document)
            # Iterate through the training vocabulary and add any log probs that match
            # if no match don't do anything. We just need a score for each category/doc
          for word, prob_cat0, prob_cat1, prob_cat2 in self.word_probs:
            if word in tokens:
              # Because of 'overflow' best to add the log probs together and exp
              log_prob_cat0 += math.log(prob_cat0)
              log_prob_cat1 += math.log(prob_cat1)
              log_prob_cat2 += math.log(prob_cat2)
            # get each of the category predictions including the prior
          cat_0_pred = self.cat_0_prior * math.exp(log_prob_cat0)
          cat_1_pred = self.cat_1_prior * math.exp(log_prob_cat1)
          cat_2_pred = self.cat_2_prior * math.exp(log_prob_cat2)
          if cat_0_pred >= cat_1_pred and cat_0_pred >= cat_2_pred:
              y_pred.append(0)
          elif cat_1_pred >= cat_0_pred and cat_1_pred >= cat_2_pred:
              y_pred.append(1)

          elif cat_2_pred >= cat_1_pred and cat_2_pred >= cat_0_pred:
              y_pred.append(2)
        return y_pred



class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        return self.mfc

                
method = sys.argv[1]
##
train_texts_fname = sys.argv[2]
train_klasses_fname = sys.argv[3]
test_texts_fname = sys.argv[4]
    
pos_words= [x.strip() for x in open('pos-words.txt',
                                           encoding='utf8')]
neg_words = [x.strip() for x in open('neg-words.txt',
                                             encoding='utf8')]
test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]

train_doc= [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
train_classes= [x.strip() for x in open(train_klasses_fname,
                                           encoding='utf8')]

if method == 'baseline':
    classifier = Baseline(train_classes)
    results = [classifier.classify(x) for x in test_texts]
    for r in results:
        print(r)

if method == 'lr':
        # Use sklearn's implementation of logistic regression
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    count_vectorizer = CountVectorizer(analyzer=tokenize)

        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
    train_counts = count_vectorizer.fit_transform(train_doc)

        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
    clf = lr.fit(train_counts, train_classes)

        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
    test_counts = count_vectorizer.transform(test_texts)
        # Predict the class for each test document
    results = clf.predict(test_counts)
    for r in results:
        print(r)

if method=="lexicon":
    test_class=baseline(pos_words,neg_words,test_texts)
    for t in test_class:
        print(t)

if method=="nb":
    nb = MNNaiveBayes()
    t=process(train_classes)
    nb.fit(train_doc,t)
    nb.check_probs()
    y_pred=nb.predict(test_texts)

    for i in y_pred:
        if i==0:
            print('negative')
        if i==1:
            print('positive')
        if i==2:
            print('neutral')

if method=="nbbin":
    nb = MNNaiveBayesBinary()
    t=process(train_classes)
    nb.fit(train_doc,t)
    nb.check_probs()
    y_pred=nb.predict(test_texts)

    for i in y_pred:
        if i==0:
            print('negative')
        if i==1:
            print('positive')
        if i==2:
            print('neutral')


