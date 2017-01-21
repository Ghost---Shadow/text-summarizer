import numpy as np
import collections
import matplotlib.pyplot as plt
import DatabaseGeneration as dg
import adjustText as at
import nltk

ITERATIONS = 1000
FILENAME = './RawText/EconomicTimes.txt'
px = [0,1,2]
py = [0,1,0]

# Labeled dataset, each word is replaced with its bag of words index
wordList,wordIndexes = dg.getDatabase(FILENAME)

n = len(wordIndexes) # Number of documents
k = 3 # Number of topics
v = len(wordList) # Number of distinct words

topicAssignment = [[int(np.random.uniform(0,k))
           for _ in range(len(wordIndexes[i]))]
           for i in range(n)]

# Dirichlet parameters
alpha = .01
lamb = .01

# How much each document pertains to a perticular topic
a = np.matrix(np.zeros((n,k)))

# How much each word pertains to a perticular topic
b = np.matrix(np.zeros((k,v)))

def rouletteArg(vector):
    #return np.argmax(vector)
    vector /= np.sum(vector)
    val = np.random.uniform()
    #print(vector)
    for i in range(len(vector)):
        val -= vector[i]
        if val <= 0:
            return i
    return len(vector)-1

def update(a,b,topicAssignment):
    # Count the number of times words from each topic are used
    # in the document
    for document in range(n):
        occurance = collections.Counter(topicAssignment[document])
        for topic in range(k):        
            a[document,topic] = occurance[topic]
        a[document] /= np.sum(a[document] + alpha)

    # Count the number of times a word is used in a particular topic
    for document in range(n):
        doc = wordIndexes[document]
        for wordIndex in range(len(doc)):
            topic = topicAssignment[document][wordIndex]
            word = wordIndexes[document][wordIndex]
            b[topic,word] += 1

    # Normalize
    for i in range(len(b.T)):
        b.T[i]/= np.sum(b.T[i] + lamb)

    # Update the assignment of topics
    for document in range(n):
        for wordIndex in range(len(topicAssignment[document])):
            vec = np.zeros(k)
            for topic in range(k):
                word = wordIndexes[document][wordIndex]
                vec[topic] = (a[document,topic] + alpha) * (b[topic,word] + lamb)
            topicAssignment[document][wordIndex] = rouletteArg(vec)

    return a,b,topicAssignment

a,b,topicAssignment = update(a,b,topicAssignment)

costs = np.zeros(ITERATIONS)
for iteration in range(ITERATIONS):
    lastA = a.copy()
    lastB = b.copy()
    a,b,topicAssignment = update(a,b,topicAssignment)
    cost = np.sum(np.abs(a-lastA)) + np.sum(np.abs(b-lastB))
    costs[iteration] = cost
    
    if cost <= 1e-7:
        break
    
    plt.figure(1)
    x = a * np.matrix(px).T
    y = a * np.matrix(py).T
    plt.plot(x,y,'bo',alpha=((1.0-iteration/ITERATIONS)*.5))

    plt.figure(2)
    x = b.T * np.matrix(px).T
    y = b.T * np.matrix(py).T
    plt.plot(x,y,'bo',alpha=((1.0-iteration/ITERATIONS)*.5))

    print(iteration,cost)

# Print the summary
chosenDocuments = np.argmax(a,0).tolist()[0]
chosenDocuments.sort()
with open(FILENAME,'r') as f:
    raw = f.read()
    sents = nltk.sent_tokenize(raw)
    for choice in chosenDocuments:
        print(sents[choice])

plt.figure(1)
fig = plt.gcf()
fig.canvas.set_window_title('Document Distribution')
x = a * np.matrix(px).T
y = a * np.matrix(py).T
plt.plot(x,y,'ro',ms=10)
plt.plot(px,py,'k-')

plt.figure(2)
fig = plt.gcf()
fig.canvas.set_window_title('Word Distribution')
x = b.T * np.matrix(px).T
y = b.T * np.matrix(py).T
plt.plot(x,y,'ro',ms=10)
yPad = (np.random.uniform() - .5) * .1
texts = []
for i in range(len(x)):
    xi = x[i].tolist()[0][0]
    yi = y[i].tolist()[0][0]
    #print(xi,yi+yPad,wordList[i])
    texts.append(plt.text(xi,yi+yPad,str(wordList[i]),fontsize=10))

at.adjust_text(texts,force_text=.05)

plt.plot(px,py,'k-')

plt.figure(3)
fig = plt.gcf()
fig.canvas.set_window_title('Cost Function')
plt.plot(costs)
plt.show()

