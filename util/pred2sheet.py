import numpy as np
import os
import os.path
import sys
import matplotlib.pyplot as plt

from math import floor

def pred2sheet(predictions_arg):
    predictions = np.array(predictions_arg)
    if predictions.shape[1] > predictions.shape[0]:
        predictions = predictions.T
    keypressed = np.zeros(predictions.shape[0]) # to see how many key pressed in a frame
    def fourier_transform(signal, period, tt): # to get the quarter duration
        f = lambda func : (signal*func(2*np.pi*tt/period)).sum()
        return f(np.cos)+ 1j*f(np.sin)

    for timeframe in range(predictions.shape[0]):
        for pitch in range(predictions.shape[1]):
            current_time = timeframe * (44 / 1000)     
            if predictions[timeframe,pitch] == 1 and predictions[timeframe-1,pitch] == 0:
                keypressed[timeframe] += 1
    
    tt = np.arange(len(keypressed))
    durations = np.arange(1.1,200,.02)
    transform = np.array([fourier_transform(keypressed,d, tt)
                        for d in durations] )
    # to know which frame has the most pressed button
    optimal_i = np.argmax(abs(transform))
    quarter_duration = durations[optimal_i]


    noteduration = []
    notedurationtemp = 0
    righthand = []
    lefthand = []
    rightchords = []
    leftchords = []
    time     = 0
    nexttime = 0
    timesincelastpress = -1
    counter = 0

    sheetnotes = ['c', 'cis', 'd', 'ees', 'e', 'f',
                'fis', 'g', 'gis', 'a', 'bes', 'b']

    sheetoctaves = [',,,',',,',',','',"'", "''", "'''"]


    for timeframe in range(predictions.shape[0]): # transcribe
        current_time = timeframe
        for pitch in range(predictions.shape[1]): # right hand
            if predictions[timeframe,pitch] == 1 and predictions[timeframe-1,pitch] == 0:
                if (timesincelastpress == -1): #first time
                    octaves = floor(((pitch+21) / 12) - 1)
                    note = round((pitch+21) % 12)
                    if(pitch+21 < 60): #if left hand
                        leftchords.append(sheetnotes[note]+sheetoctaves[octaves])
                        rightchords.append("r")
                    else: #if right hand
                        rightchords.append(sheetnotes[note]+sheetoctaves[octaves])
                        leftchords.append("r")
                    # chords.append(pitch+21)
                    timesincelastpress = current_time
                elif((current_time - timesincelastpress) < (quarter_duration / 4)):
                    octaves = floor(((pitch+21) / 12) - 1)
                    note = round((pitch+21) % 12)
                    if(pitch+21 < 60): #if left hand
                        leftchords.append(sheetnotes[note]+sheetoctaves[octaves])
                        rightchords.append("r")
                    else: #if right hand
                        rightchords.append(sheetnotes[note]+sheetoctaves[octaves])
                        leftchords.append("r")
                    # chords.append(pitch+21)
                elif((current_time - timesincelastpress) > (quarter_duration / 4)):
                    notedurationtemp = round((current_time - timesincelastpress) / (quarter_duration / 4))
                    if (notedurationtemp == 1):
                        noteduration.append("16")
                    elif (notedurationtemp == 2 or notedurationtemp == 3):
                        noteduration.append("8")
                    elif (notedurationtemp == 4 or notedurationtemp == 5):
                        noteduration.append("4")
                    elif (notedurationtemp == 6 or notedurationtemp == 7):
                        noteduration.append("4.")
                    elif (notedurationtemp == 8 or notedurationtemp == 9 or notedurationtemp == 10):
                        noteduration.append("2")  
                    elif (notedurationtemp == 11 or notedurationtemp == 12 or notedurationtemp == 13 or notedurationtemp == 14):
                        noteduration.append("2.")
                    else:
                        noteduration.append("1")  

                    righthand.append(np.array(rightchords))
                    lefthand.append(np.array(leftchords))
                    rightchords.clear()
                    leftchords.clear()
                    octaves = floor(((pitch+21) / 12) - 1)
                    note = round((pitch+21) % 12)
                    if(pitch+21 < 60): #if left hand
                        leftchords.append(sheetnotes[note]+sheetoctaves[octaves])
                        rightchords.append("r")
                    else: #if right hand
                        rightchords.append(sheetnotes[note]+sheetoctaves[octaves])
                        leftchords.append("r")
                    # chords.append(pitch+21)
                    timesincelastpress = current_time

        if(timeframe == (predictions.shape[0]-1) and len(rightchords)!=0): #last frame
            notedurationtemp = round((current_time - timesincelastpress) / (quarter_duration / 4))
            if (notedurationtemp == 1):
                noteduration.append("16")
            elif (notedurationtemp == 2 or notedurationtemp == 3):
                noteduration.append("8")
            elif (notedurationtemp == 4 or notedurationtemp == 5):
                noteduration.append("4")
            elif (notedurationtemp == 6 or notedurationtemp == 7):
                noteduration.append("4.")
            elif (notedurationtemp == 8 or notedurationtemp == 9 or notedurationtemp == 10):
                noteduration.append("2")  
            elif (notedurationtemp == 11 or notedurationtemp == 12 or notedurationtemp == 13 or notedurationtemp == 14):
                noteduration.append("2.")
            else:
                noteduration.append("1")  

            if(pitch+21 < 60): #if left hand
                leftchords.append(sheetnotes[note]+sheetoctaves[octaves])
                rightchords.append("r")
            else: #if right hand
                rightchords.append(sheetnotes[note]+sheetoctaves[octaves])
                leftchords.append("r")

    #//////////////////write to file
    # filename = "test.ly"

    # with open(filename, 'w+') as f:
    #     f.write("\\version \"2.20.0\"\score{\\new PianoStaff = \"piano\"<<\\new staff = \"upper\"{ \\tempo 4=240"
    #             + "\n")
    #     #print right hand
    #     i = 0
    #     for chord in righthand:
    #         restcounter = 0
    #         f.write("<")
    #         for key in chord:
    #             if(key != "r"):
    #                 f.write("%s "%key)
    #             else:
    #                 restcounter += 1
    #         f.write(">"+noteduration[i])
    #         if(restcounter != 0):
        
    #                 f.write("r%s "%noteduration[i])        
    #         i += 1
    #         if(i>300):
    #             break
    #     f.write("}")
    #     #print left hand
    #     f.write("\\new Staff = \"lower\" {\\tempo 4 = 240\\clef bass"
    #             + "\n")
    #     i = 0
    #     for chord in lefthand:
    #         restcounter = 0
    #         f.write("<")
    #         for key in chord:
    #             if(key != "r"):
    #                 f.write("%s "%key)
    #             else:
    #                 restcounter += 1
    #         f.write(">"+noteduration[i])
    #         if(restcounter != 0):
    #                 f.write("r%s "%noteduration[i])        
    #         i += 1
    #         if(i>300):
    #             break
    #     f.write("}>>\layout { }\midi { }}")

    #//////////////////////////////////////////////////////////////


    notations = ("\\version \"2.20.0\"\score{\\new PianoStaff = \"piano\"<<\\new staff = \"upper\"{ \\tempo 4=240"
            + "\n")
    #print right hand
    i = 0
    for chord in righthand:
        restcounter = 0
        notations = notations + ("<")
        for key in chord:
            if(key != "r"):
                notations = notations + ("%s "%key)
            else:
                restcounter += 1
        notations = notations + (">"+noteduration[i])
        if(restcounter != 0):

                notations = notations + ("r%s "%noteduration[i])        
        i += 1
        if(i>300):
            break
    notations = notations + ("}")
    #print left hand
    notations = notations + ("\\new Staff = \"lower\" {\\tempo 4 = 240\\clef bass"
            + "\n")
    i = 0
    for chord in lefthand:
        restcounter = 0
        notations = notations + ("<")
        for key in chord:
            if(key != "r"):
                notations = notations + ("%s "%key)
            else:
                restcounter += 1
        notations = notations + (">"+noteduration[i])
        if(restcounter != 0):
                notations = notations + ("r%s "%noteduration[i])        
        i += 1
        if(i>300):
            break
    notations = notations + ("}>>\layout { }\midi { }}")

    return notations