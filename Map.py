import tkinter as tk
import os
import time
import copy

# Define the main window
top = tk.Tk()
top.title("Intelligent Gomoku Game") # Set the window title
top.geometry('400x500') # Set the window size

# Define the size of the game map
mapsize = 8

# Define the size of each element on the map
pixsize = 20

# Define the number of consecutive pieces required to win
winSet = 5

# Define the codes for different states on the map
# Code for empty space
backcode = 0
# Code for white pieces
whitecode = 1
# Code for black pieces
blackcode = -1

# Create the canvas for drawing the game map
canvas = tk.Canvas(top, height=mapsize * pixsize, width=mapsize * pixsize,
                 bg = "gray")  # Set canvas size and background color
canvas.pack(pady=10)  # Add padding to the canvas

# Draw the grid lines on the canvas
for i in range(mapsize):
    canvas.create_line(i * pixsize, 0,
                  i * pixsize, mapsize * pixsize,
                  fill='black')  # Draw vertical lines
    canvas.create_line(0, i * pixsize,
                  mapsize * pixsize, i * pixsize,
                  fill='black')  # Draw horizontal lines

# Initialize the game boards
whiteBoard = []
stepBoard = []
for i in range(mapsize):
    row = []
    rowBak = []
    for j in range(mapsize):
        row.append(0)
        rowBak.append(backcode)
    whiteBoard.append(rowBak)
    stepBoard.append(row)
blackBoard = copy.deepcopy(whiteBoard)

# List to store the drawn pieces on the canvas
childMap = []

# Lists to record the game states and steps
mapRecords1 = []
mapRecords2 = []
stepRecords1 = []
stepRecords2 = []

# Score records
scoreRecords1 = []
scoreRecords2 = []

isGameOver = False

IsTurnWhite = True
# Function to restart the game
def Restart():
    global isGameOver
    global IsTurnWhite
    for child in childMap:
        canvas.delete(child)
    childMap.clear()
    isGameOver = False
    IsTurnWhite = True
    mapRecords1.clear()
    mapRecords2.clear()
    stepRecords1.clear()
    stepRecords2.clear()
    scoreRecords1.clear()
    scoreRecords2.clear()
    for i in range(mapsize):
        for j in range(mapsize):
            whiteBoard[j][i] = backcode
            blackBoard[j][i] = backcode

# Define paths for saving datasets
WinDataSetPath = 'DataSets\\win'
LosDataSetPath = 'DataSets\\los'

TrainNet = None

# Function to save game data to files
def SaveDataSet(tag):
    if TrainNet != None:
        TrainNet(tag)
    else:
        winfilename = WinDataSetPath + '\\' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt'
        losfilename = LosDataSetPath + '\\' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt'
        if not os.path.exists('DataSets'):
            os.mkdir('DataSets')
        if not os.path.exists(WinDataSetPath):
            os.mkdir(WinDataSetPath)
        if not os.path.exists(LosDataSetPath):
            os.mkdir(LosDataSetPath)
        strInfo1 = ''
        for i in range(len(mapRecords1)):
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo1 += str(mapRecords1[i][j][k]) + ','
            strInfo1 += '\n'
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo1 += str(stepRecords1[i][j][k]) + ','
            strInfo1 += '\n'
        strInfo2 = ''
        for i in range(len(mapRecords2)):
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo2 += str(mapRecords2[i][j][k]) + ','
            strInfo2 += '\n'
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo2 += str(stepRecords2[i][j][k]) + ','
            strInfo2 += '\n'
        if tag == 1:
            with open(winfilename,"w") as f:
                f.write(strInfo1)
            with open(losfilename,"w") as f:
                f.write(strInfo2)
        else:
            with open(winfilename,"w") as f:
                f.write(strInfo2)
            with open(losfilename,"w") as f:
                f.write(strInfo1)
           
# Function to check if the game is over and determine the winner
def JudgementResult():
    global isGameOver
    judgemap = whiteBoard
    for i in range(mapsize):
        for j in range(mapsize):
            if judgemap[j][i] != backcode:
                tag = judgemap[j][i]
                checkrow = True
                checkCol = True
                checkLine = True
                checkLine2 = True
                for k in range(winSet - 1):
                    if i + k + 1 < mapsize: # Check row
                        if (judgemap[j][i + k + 1] != tag) and checkrow:
                            checkrow = False
                        if j + k + 1 < mapsize: # Check diagonal
                            if (judgemap[j + k + 1][i + k + 1] != tag) and checkLine:
                               checkLine = False
                        else:
                            checkLine = False
                    else:
                        checkrow = False
                        checkLine = False
                    if j + k + 1 < mapsize: # Check column
                       if (judgemap[j + k + 1][i] != tag) and checkCol:
                           checkCol = False
                       if i - k - 1 >= 0: # Check diagonal
                            if (judgemap[j + k + 1][i - k - 1] != tag) and checkLine2:
                               checkLine2 = False
                       else:
                            checkLine2 = False
                    else:
                        checkCol = False
                        checkLine2 = False
                    if not checkrow and not checkCol and not checkLine and not checkLine2:
                        break
                if checkrow or checkCol or checkLine or checkLine2:
                    if AutoPlay == 0:
                        print (str(tag) + 'win')
                        print ('game over!')
                    isGameOver = True
                    SaveDataSet(tag)
                    return tag
    return 0
            
PlayWithComputer = None

AutoPlay = 0

GetMaxScore = None

# Function to handle user's mouse click event
def playChess(event):
    global AutoPlay
    if isGameOver:
        print('game is over, restart!')
        Restart()
        return 
    x = event.x // pixsize
    y = event.y // pixsize
    if x >= mapsize or y >= mapsize:
        return
    if whiteBoard[y][x] != backcode:
        return   
    score = 0
    if PlayWithComputer != None:
        _x, _y, score = PlayWithComputer(IsTurnWhite)
    res = chess(x, y, score)
    if res == 0:
        if PlayWithComputer != None:
            x, y, score = PlayWithComputer(IsTurnWhite)
            res = chess(x,y,score)
            while AutoPlay > 0:
                while res == 0:
                    x, y, score = PlayWithComputer(IsTurnWhite)
                    res = chess(x,y,score)
                AutoPlay -= 1
                chess(x,y,score)
                x, y, score = PlayWithComputer(IsTurnWhite)
                res = chess(x,y,score)
    
# Function to place a chess piece on the board   
def chess(x,y,score):
    global IsTurnWhite
    if isGameOver:
        if AutoPlay == 0:
            print('game is over, restart!')
        Restart()
        return -1
    if whiteBoard[y][x] != backcode:
        if AutoPlay == 0:
            print('game is over, restart!')
        Restart()
        return -1    
    step = copy.deepcopy(stepBoard)
    step[y][x] = 1
    if IsTurnWhite: # If it's white's turn
        mapRecords1.append(copy.deepcopy(blackBoard))
        stepRecords1.append(step)
        scoreRecords1.append(score)
        whiteBoard[y][x] = whitecode # 1 white -1 black
        blackBoard[y][x] = blackcode
        child = canvas.create_oval(x * pixsize,
                                   y * pixsize, 
                                   x * pixsize + pixsize,  
                                   y * pixsize + pixsize, fill='white')
    else:  # If it's black's turn
        mapRecords2.append(copy.deepcopy(whiteBoard))
        stepRecords2.append(step)
        scoreRecords2.append(score)
        whiteBoard[y][x] = blackcode # 1 white -1 black
        blackBoard[y][x] = whitecode
        child = canvas.create_oval(x * pixsize,
                                   y * pixsize, 
                                   x * pixsize + pixsize,  
                                   y * pixsize + pixsize, fill='black')
    IsTurnWhite = not IsTurnWhite        
    childMap.append(child)
    return JudgementResult()

# Function to increase the autoplay count by 1000
def ReAutoPlay():
    global AutoPlay
    AutoPlay += 1000
 
btnUp = tk.Button(top, text ="Auto-Training 1000times", command = ReAutoPlay)
btnUp.pack()

# Function to make one autoplay move
def AutoPlayOnce():
    if PlayWithComputer != None:
        x, y, score = PlayWithComputer(IsTurnWhite)
        chess(x,y,score)

# Create a button to trigger AutoPlayOnce function 
btnAuto = tk.Button(top, text ="Auto-Move once", command = AutoPlayOnce)
btnAuto.pack()


# Bind the canvas to the left mouse button click event
#canvas.bind("<B1-Motion>", playChess)
canvas.bind("<Button-1>", playChess)

# Function to display the game window
def ShowWind():
    top.mainloop()