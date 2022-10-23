# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:48:28 2022

@author: sirco
"""

from tkinter import *
import csv

def makeEmptySheet(r, w) :
    retList = []
    for i in range(0, r):
        tmpList = []
        for k in range(0, w):
            ent = Entry(window, text='', width=10)
            ent.grid(row=i, column=k)
            tmpList.append(ent)
        retList.append(tmpList)
    return retList

csvList = []
rowNum, colNum = 0, 0
workSheet = []

window = Tk()

with open("./singer1.csv", "r") as inFp :
    csvReader = csv.reader(inFp)
    header_list = next(csvReader)
    csvList.append(header_list)
    for row_list in csvReader:
        csvList.append(row_list)

    rowNum = len(csvList)
    colNum = len(csvList[0])
    workSheet = makeEmptySheet(rowNum, colNum)

    for i in range(0, rowNum) :
        for k in range(0, colNum) :
            if ( k == 2 and csvList[i][k].isnumeric() ) :
                if ( int(csvList[i][k]) >= 7) :
                    ent = workSheet[i][k]
                    ent.configure(bg='magenta')
            
            workSheet[i][k].insert(0, csvList[i][k])
            
            
window.mainloop()