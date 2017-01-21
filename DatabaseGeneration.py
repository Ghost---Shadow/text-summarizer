import nltk

def getDatabase(fileName):
    with open(fileName,'r') as f:
        raw = f.read()

    sentences = nltk.sent_tokenize(raw)
    wordList = list(set(nltk.word_tokenize(raw)))
    wordIndexes = []
    for sentence in sentences:
        #print(sentence)
        words = nltk.word_tokenize(sentence)
        wordIndexes.append([])
        for word in words:
            wordIndexes[-1].append(wordList.index(word))

    return wordList,wordIndexes

def reconstruct(wordList,wordIndexes):
    for sentence in wordIndexes:
        s = ''
        for word in sentence:
            s += wordList[word]+' '
        print(s)

#a,b = getDatabase('./RawText/EconomicTimes.txt')
#reconstruct(a,b)
