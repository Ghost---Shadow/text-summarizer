import nltk
import numpy as np

raw_sentences = list(nltk.corpus.gutenberg.sents("burgess-busterbrown.txt"))
print(len(raw_sentences))
sentences = []

# Drop small sentences
for sentence in raw_sentences:
    if len(sentence) > 6:
        sentences.append(sentence)
        
sentences = sentences[:400]
sentCount = len(sentences)
print(sentCount)

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

c = 0
for sentences in paragraphs:
    sentenceRank = [0 for _ in range(PARAGRAPH_SIZE)]    
    
    for i in range(PARAGRAPH_SIZE):        
        for j in range(PARAGRAPH_SIZE):
            if i != j:
                sentenceRank[i] += intersection(sentences[i],sentences[j])
                
    chosen = int(np.argmax(sentenceRank))
    print(' '.join(sentences[chosen]))
    #print(str(max(sentenceRank)) +' '+ str(c))
    c+=1


