# -*- coding: UTF-8 -*-
import re


def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


if __name__ == '__main__':
    docList = []; classList = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    print(vocabList)