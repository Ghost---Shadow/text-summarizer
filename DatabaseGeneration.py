import nltk

def getDatabase(fileName,onlyNouns = True,keepUNK = False):
    with open(fileName,'r') as f:
        raw = f.read()

    # Get all sentences
    sentences = nltk.sent_tokenize(raw)

    # Generate bag of words
    wordList = nltk.word_tokenize(raw)

    # Drop all non-nouns
    tagged = nltk.pos_tag(wordList)
    wordList = []
    for word,pos in tagged:
        if pos[0] == 'N':
            wordList.append(word)

    wordList = list(set(wordList))
    wordList = ["UNK"] + wordList
    wordIndexes = []
    for sentence in sentences:
        #print(sentence)
        words = nltk.word_tokenize(sentence)
        wordIndexes.append([])
        for word in words:
            if word in wordList:
                wordIndexes[-1].append(wordList.index(word))
            else:
                if keepUNK:
                    wordIndexes[-1].append(0) # Unknown

    return wordList,wordIndexes

def reconstruct(wordList,wordIndexes):
    for sentence in wordIndexes:
        s = ''
        for word in sentence:
            s += wordList[word]+' '
        print(s)

#a,b = getDatabase('./RawText/EconomicTimes.txt')
#reconstruct(a,b)
