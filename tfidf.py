import re
import numpy as np
def load():
    docdata = []
    label = []
    for i in range(1,26):
        # class 1
        onestring = open("email/spam/%d.txt" % i).read()
        strarr = re.split(r'\W*', onestring)
        wordlist = [word.lower() for word in strarr if len(word) > 2]
        docdata.append(wordlist)
        label.append(1)
        # class 0
        onestring = open("email/ham/%d.txt" % i).read()
        strarr = re.split(r'\W*', onestring)
        wordlist = [word.lower() for word in strarr if len(word) > 2]
        docdata.append(wordlist)
        label.append(0)
    return docdata, label
def bayes():
    print("bayes")
    docdata, label = load()
    # create vocabulary 
    vocab = set([])
    for doc in docdata:
        vocab = vocab | set(doc)
    vocab = list(vocab)
    d = len(vocab)
    n = len(docdata)
    tf = []
    idf = []
    # tf(w,d) = count(w,d) / size(d) ; w:word in d:document
    # idf = log(n/ docs(w,D)) n=docs number, docs(w,D)=docs number contain w
    for doc in docdata:
        one = [0] * d
        onef = [0] * d
        for word in doc:
            one[vocab.index(word)] += 1.0/len(doc)
            onef[vocab.index(word)] = 1
        tf.append(one)
        idf.append(onef)
    tf = np.array(tf)
    idf = np.array(idf)
    idf = np.tile(np.log(n / np.sum(idf, axis=0)).reshape(1,-1), (n,1))
    tfidf = tf*idf
    #windx = np.argsort(-tfidf[0])
    #print(np.array(vocab)[windx[0:10]])
if __name__ == "__main__":
    bayes()
