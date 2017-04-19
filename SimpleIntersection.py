import nltk
import numpy as np

raw_sentences = []
#raw_sentences = list(nltk.corpus.gutenberg.sents("burgess-busterbrown.txt"))
raw = ""
FILENAME = './RawText/EconomicTimes.txt'
#FILENAME = './RawText/Multidocument.txt'

with open(FILENAME,'r') as f:
    raw = f.read()
    
for sent in nltk.sent_tokenize(raw):
    words = nltk.word_tokenize(sent)
    raw_sentences.append(words)
    
print(len(raw_sentences))
sentences = []

# Drop small sentences
for sentence in raw_sentences:
    if len(sentence) > 6:
        sentences.append(sentence)
        
#sentences = sentences[:400]
sentCount = len(sentences)
print('Number of sentences:',sentCount)

# Convert to lowercase and drop punctuation
for i in range(sentCount):
    words = [w.lower() for w in sentences[i] if w.isalnum()]
    sentences[i] = words

PARAGRAPH_SIZE = 5
paragraphs = [sentences[i:i+PARAGRAPH_SIZE] for i in range(0,sentCount,PARAGRAPH_SIZE)]
#  f(s1, s2) = |{w | w in s1 and w in s2}| / ((|s1| + |s2|) / 2)
def intersection(sent1,sent2):    
    words1 = set(sent1)
    words2 = set(sent2)
    avgLen = (len(words1)+len(words2))/2.0
    
    return len(words1.intersection(words2))/avgLen

for paragraph in range(len(paragraphs)):
    sentences = paragraphs[paragraph]
    sentenceRank = [0 for _ in range(PARAGRAPH_SIZE)]    
    
    for i in range(len(sentences)):        
        for j in range(len(sentences)):
            if i != j:
                sentenceRank[i] += intersection(sentences[i],sentences[j])
                
    choice = int(np.argmax(sentenceRank))
    #print(choice,' '.join(sentences[choice]))
    index = paragraph*PARAGRAPH_SIZE+choice
    #print(paragraph,PARAGRAPH_SIZE,choice)
    print(index,' '.join(raw_sentences[index]))
