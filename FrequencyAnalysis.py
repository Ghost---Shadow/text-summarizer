import nltk
import numpy as np
import matplotlib.pyplot as plt
'''
from nltk.corpus import brown
rawWords = brown.words(fileids=['ca02'])
rawSentences = brown.sents(fileids=['ca02'])
'''
'''
from nltk.corpus import gutenberg
rawWords = gutenberg.words(fileids=['burgess-busterbrown.txt'])
rawSentences = gutenberg.sents(fileids=['burgess-busterbrown.txt'])
'''
#FILENAME = './RawText/EconomicTimes.txt'
#FILENAME = './RawText/Multidocument.txt'
FILENAME = './RawText/1/raw.txt'
OUTPUT = './RawText/1/S_Freq.txt'

raw = ""
with open(FILENAME,'r') as f:
    raw = f.read()

rawWords = []
rawSentences = []
for sent in nltk.sent_tokenize(raw):
    words = nltk.word_tokenize(sent)
    rawWords.extend(words)
    rawSentences.append(words)

MIN_SENTENCE_LENGTH = 5

# Remove punctuations
def clean(words):
    newWords = []    
    for word in words:
        if word.isalnum():
            newWords.append(word.lower())
    return newWords

words = clean(rawWords)
wordCount = len(words)
DROP_FREQ = int(wordCount * .01)

sentences = []
# Clean sentences
for sent in rawSentences:
    sent = clean(sent)
    if len(sent) > MIN_SENTENCE_LENGTH:
        sentences.append(sent)
    else:
        sentences.append([])

NUMBER_OF_LINES_TO_EXTRACT = 5

print("Total sentences:",len(sentences))

# Count the frequency of words
distribution = nltk.FreqDist(words)

# Remove the most frequent words like, 'the','a','in' etc
for _ in range(DROP_FREQ):
    key = max(distribution,key=distribution.get)
    print('Dropping',key)
    distribution[key] = 0

# Normalize the distribution
for word in distribution:
    distribution[word] = distribution[word]/wordCount

# Score(S) = sigma(p(w), for w belongs to S) / |S|
def getSentenceScore(sent,dist):
    score = 0
    validWords = 0
    for word in sent:        
        score += dist[word.lower()]
        validWords += 1
    
    return score/validWords

# Rank all the sentences
def rankAllSentences(sents,dist):
    ranks = []
    for sent in sents:
        if len(sent) != 0:
            ranks.append(getSentenceScore(sent,dist))
        else:
            ranks.append(0)
    return ranks

# Reduce weight of words that have been picked
def updateWordScore(sent,dist):
    for word in sent:
        dist[word] = dist[word] ** 2

choices = set()
for iteration in range(NUMBER_OF_LINES_TO_EXTRACT):
    ranks = rankAllSentences(sentences,distribution)
    choice = np.argmax(ranks)
    choices.add(choice)
    updateWordScore(sentences[choice],distribution)
    #sentences[choice] = []

    # Graphing
    wordScore = []
    for key in distribution:
        wordScore.append(distribution[key])
    v = np.zeros_like(ranks)
    v[choice] = max(ranks)
    plt.figure(1)
    plt.plot(ranks,str((1.0-iteration/NUMBER_OF_LINES_TO_EXTRACT)*.5))
    plt.plot(v,'ro')
    plt.figure(2)
    plt.plot(wordScore,str((1.0-iteration/NUMBER_OF_LINES_TO_EXTRACT)*.5))

# Print the summary
choices = list(choices)
choices.sort()
dump = ''
for choice in choices:
    line = " ".join(rawSentences[choice])
    dump += line
    print(choice,line)

with open(OUTPUT,'w') as f:
    f.write(dump)

plt.show()
