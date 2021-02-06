# -*- coding: utf-8 -*-

def getOffset(tagset_matrix, ntag=4):
    offset_matrix = []
    for index, sentence in enumerate(tagset_matrix):
        offset = []
        start = 0
        end = 0
        for i, tag in enumerate(sentence):
            if ntag == 4:
                if tag == 2 or i == (len(sentence)-1):
                    offset.append([start,end])
                    start = i + 1
                    end = i + 1
                elif tag == 4 or tag == 0:
                    if start != end:
                        offset.append([start, end-1])
                    offset.append([end,end])                
                    start = i + 1
                    end = i + 1
                elif tag == 1:
                    if start != end:
                        offset.append([start, end-1])
                        start = i
                    end += 1
                elif tag == 3:
                    end += 1
            elif ntag == 2:
                if tag == 2 or tag == 0 or i == (len(sentence)-1):
                    offset.append([start, end])
                    start = i + 1
                    end = i + 1        
                elif tag == 1:
                    end += 1
        offset_matrix.append(offset)
    return offset_matrix
    
def calculate(lineOffsetGold, lineOffsetResult, dataset):
    sameToken = 0
    goldToken = 0
    resultToken = 0
    sizeG = len(lineOffsetGold)
    sizeR = len(lineOffsetResult)
    
    if sizeG != sizeR:
        print('size err', sizeG, sizeR)
        return 0
    for index in range(sizeG):
        lineG = lineOffsetGold[index]
        lineR = lineOffsetResult[index]
        sizelineG = len(lineG)
        sizelineR = len(lineR)
        indexG = 0
        indexR = 0
        while (indexG < sizelineG and indexR < sizelineR):
            startG = lineG[indexG][0]
            endG = lineG[indexG][1]
            startR = lineR[indexR][0]
            endR = lineR[indexR][1]           
            if startG == startR and endG == endR:
                sameToken += 1
                indexG += 1
                indexR += 1
            elif endG > endR:
                resultToken += 1
                indexR += 1
            elif endR > endG:
                goldToken += 1
                indexG += 1
            else:
                resultToken += 1
                goldToken += 1
                indexG += 1
                indexR += 1
        if indexG < sizelineG:
            goldToken = goldToken+sizelineG-1-indexG
        elif indexR < sizelineR:
            resultToken = resultToken+sizelineR-1-indexR
    if dataset == 'pku':
        wtoken = 16044
    elif dataset == 'msr':
        wtoken = 17204
    elif dataset == 'cityu':
        wtoken = 6285
    elif dataset == 'as':
        wtoken = 19443
    P = float(sameToken + wtoken)/float(sameToken + resultToken + wtoken)
    R = float(sameToken + wtoken)/float(sameToken + goldToken + wtoken)
    F = 2*P*R/(P+R)
    P = round(P, 6)
    R = round(R, 6)
    F = round(F, 6)
    return P, R, F

def accuracy(result, gold):
    tagToken = 0
    sameToken = 0
    for i in range(len(result)):
        result_sent = result[i]
        gold_sent = gold[i]
        for index, tag in enumerate(result_sent):
            tagToken += 1
            if tag == gold_sent[index]:
                sameToken += 1
    acc = float(sameToken) / float(tagToken)
    acc = round(acc, 6)
    return acc*100

def score(result, gold):
    #result为分词结果列表，gold为分词答案列表
    #result/gold = [[],[],[]];
    result_offset = getOffset(result)
    gold_offset = getOffset(gold)
    P, R, F = calculate(gold_offset, result_offset, 'pku')
    return P*100, R*100, F*100
