import pygame
from math import floor
import time
from copy import copy
import tensorflow as tf
from collections import deque
import numpy as np
import random
from datetime import datetime



# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=100)
        self.gamma = 0.8
        self.epsilon = 0.9999
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.20
        self.batch_size = 500

        self.learning_rate =0.0000000001

        self.model = self.build_model()



    def evaulate(self, list_of_states, whos_turn):
        value_list = []
        random = np.random.rand()
        can_finish = False
        list_index = 0
        finish_index = 99
        for state in list_of_states:
            if 6 not in state:
                finish_state = state
                finish_index = list_index
                can_finish = True
            elif -6 not in state:
                finish_state = state
                finish_index = list_index
                can_finish = True

            state = np.asarray(state)
            # state = state / 6
            state = state.reshape(8,8,1)

            state = np.array([state])

            value_state =self.model.predict(state)
            value_list.append(value_state)
            list_index += 1

        value_array = np.asarray(value_list)


        if whos_turn == 'W':
            preferred_index = np.argmax(value_array)
            preferred_state = list_of_states[preferred_index]
        elif whos_turn == 'B':
            preferred_index = np.argmin(value_array)
            preferred_state = list_of_states[preferred_index]

        if random <= self.epsilon:
            preferred_index = np.random.randint(0,len(value_array))
            preferred_state = list_of_states[preferred_index]

        if preferred_index == finish_index:
            print('corect')

        if can_finish:
            loss_value = abs(abs(value_list[preferred_index]) - abs(value_list[finish_index]))
            print("loss is = " + str(loss_value))
            preferred_state = finish_state
            preferred_index = finish_index






        return value_list, preferred_state, preferred_index

    def build_model(self):



        activation_leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.5)
        activation_softmax = tf.keras.layers.Softmax()
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                                         input_shape=(8, 8, 1)))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)


        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)

        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)

        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)

        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)

        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(activation_leakyRelu)
        model.add(tf.keras.layers.Dense(1))





        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))


        return model

    def remember(self, state, reward, next_state, done):
        self.memory.append((state, reward, next_state, done))

    def replay(self, batch_size):

        # logdir = "logs/" + modelName
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        minibatch = random.sample(self.memory, batch_size)
        sure_reward = self.memory[-1]
        minibatch.append(sure_reward)

        random.shuffle(minibatch)

        for state, reward, next_state, done in minibatch:
            target = reward
            state_list = []
            next_state_list = []
            for key_state in state.keys():
                state_list.append(state[key_state])

            for key_next_state in next_state.keys():
                next_state_list.append(next_state[key_next_state])

            state = state_list
            next_state = next_state_list

            state = np.asarray(state)
            # state = state / 6
            state = state.reshape(8, 8, 1)
            state = np.array([state])

            next_state = np.asarray(next_state)
            # next_state = next_state / 6
            next_state = next_state.reshape(8, 8, 1)
            next_state = np.array([next_state])

            if not done:
                target = (reward + self.gamma * self.model.predict(next_state))
            self.model.fit(state, [target], epochs=1, verbose=0)
            # Decrease epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay





def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key

    return -1


def roundDown(value):
    return int(floor(value / 100) * 100)


class GameEnv:

    def __init__(self):

        pygame.init()
        self.WINSIZE = 800
        self.sizeBetween = self.WINSIZE / 8

        self.coordinateDict = self.CreateLocationDict()
        self.selectedPiece = None
        self.selectedPos = []
        self.selectedPieceCoord = None
        self.isSelected = False
        self.whosTurn = 'W'
        self.possibleStateW = {}
        self.possibleStateB = {}
        self.selectedPieceNotation = None
        self.blocked = True
        self.allPossibleForSelected = {}
        self.notationDict = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h'}
        self.reward = 0
        # Setting Sprites

        self.Wpawn = pygame.image.load(r'pieces/WhitePawn.png')
        self.Wrook = pygame.image.load(r'pieces/WhiteRook.png')
        self.Wknight = pygame.image.load(r'pieces/WhiteKnight.png')
        self.Wbishop = pygame.image.load(r'pieces/WhiteBishop.png')
        self.Wqueen = pygame.image.load(r'pieces/WhiteQueen.png')
        self.Wking = pygame.image.load(r'pieces/WhiteKing.png')

        self.Bpawn = pygame.image.load(r'pieces/BlackPawn.png')
        self.Brook = pygame.image.load(r'pieces/BlackRook.png')
        self.Bknight = pygame.image.load(r'pieces/BlackKnight.png')
        self.Bbishop = pygame.image.load(r'pieces/BlackBishop.png')
        self.Bqueen = pygame.image.load(r'pieces/BlackQueen.png')
        self.Bking = pygame.image.load(r'pieces/BlackKing.png')

        # Scaling Sprites

        self.Wpawn = pygame.transform.scale(self.Wpawn, (100, 100))
        self.Wrook = pygame.transform.scale(self.Wrook, (100, 100))
        self.Wknight = pygame.transform.scale(self.Wknight, (100, 100))
        self.Wbishop = pygame.transform.scale(self.Wbishop, (100, 100))
        self.Wqueen = pygame.transform.scale(self.Wqueen, (100, 100))
        self.Wking = pygame.transform.scale(self.Wking, (100, 100))

        self.Bpawn = pygame.transform.scale(self.Bpawn, (100, 100))
        self.Brook = pygame.transform.scale(self.Brook, (100, 100))
        self.Bknight = pygame.transform.scale(self.Bknight, (100, 100))
        self.Bbishop = pygame.transform.scale(self.Bbishop, (100, 100))
        self.Bqueen = pygame.transform.scale(self.Bqueen, (100, 100))
        self.Bking = pygame.transform.scale(self.Bking, (100, 100))

        # Positions
        self.mySelection = []
        # (keys for position dict)

        # Pawn
        self.WpawnPositions = ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']
        self.BpawnPositions = ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7']

        # Rook
        self.WrookPositions = ['a1', 'h1']
        self.BrookPositions = ['a8', 'h8']

        # Knight
        self.WknightPositions = ['b1', 'g1']
        self.BknightPositions = ['b8', 'g8']

        # Bishop
        self.WbishopPositions = ['c1', 'f1']
        self.BbishopPositions = ['c8', 'f8']

        # Queen
        self.WqueenPositions = ['d1']
        self.BqueenPositions = ['d8']

        # King
        self.WkingPositions = ['e1']
        self.BkingPositions = ['e8']

        self.valueNotationDict = self.getValueNotation()

    def get_output(self):
        mystates={}
        if self.whosTurn == 'W':
            mystates = self.possibleStateW
        elif self.whosTurn =='B':
            mystates = self.possibleStateB

        return mystates

    def get_state(self):
        board = copy(self.boardDict)
        return board
    #
    # @staticmethod
    # def state_to_binstate(state):
    #



    def board_to_position(self, board_dict):
        self.WpawnPositions = []
        self.BpawnPositions = []
        # Rook
        self.WrookPositions = []
        self.BrookPositions = []
        # Knight
        self.WknightPositions = []
        self.BknightPositions = []
        # Bishop
        self.WbishopPositions = []
        self.BbishopPositions = []
        # Queen
        self.WqueenPositions = []
        self.BqueenPositions = []
        # King
        self.WkingPositions = []
        self.BkingPositions = []

        for notation in board_dict.keys():

            if board_dict[notation] == 1:
                self.WpawnPositions.append(notation)
            elif board_dict[notation] == 2:
                self.WrookPositions.append(notation)
            elif board_dict[notation] == 3:
                self.WknightPositions.append(notation)
            elif board_dict[notation] == 4:
                self.WbishopPositions.append(notation)
            elif board_dict[notation] == 5:
                self.WqueenPositions.append(notation)
            elif board_dict[notation] == 6:
                self.WkingPositions.append(notation)
            elif board_dict[notation] == -1:
                self.BpawnPositions.append(notation)
            elif board_dict[notation] == -2:
                self.BrookPositions.append(notation)
            elif board_dict[notation] == -3:
                self.BknightPositions.append(notation)
            elif board_dict[notation] == -4:
                self.BbishopPositions.append(notation)
            elif board_dict[notation] == -5:
                self.BqueenPositions.append(notation)
            elif board_dict[notation] == -6:
                self.BkingPositions.append(notation)
            pass

    def TurnOver(self):
        if self.whosTurn == 'W':
            self.whosTurn ='B'
        elif self.whosTurn == 'B':
            self.whosTurn = 'W'


    def getValueNotation(self):

        valueNotation = {1: [self.WpawnPositions, 'Wpawn', self.Wpawn], -1: [self.BpawnPositions, 'Bpawn', self.Bpawn],
                         2: [self.WrookPositions, 'Wrook', self.Wrook], -2: [self.BrookPositions, 'Brook', self.Brook],
                         3: [self.WknightPositions, 'Wknight', self.Wknight],
                         -3: [self.BknightPositions, 'Bknight', self.Bknight],
                         4: [self.WbishopPositions, 'Wbishop', self.Wbishop],
                         -4: [self.BbishopPositions, 'Bbishop', self.Bbishop],
                         5: [self.WqueenPositions, 'Wqueen', self.Wqueen],
                         -5: [self.BqueenPositions, 'Bqueen', self.Bqueen],
                         6: [self.WkingPositions, 'Wking', self.Wking], -6: [self.BkingPositions, 'Bking', self.Bking]}

        return valueNotation

    def getPieces(self, window):

        self.drawPieces(window, self.WpawnPositions, self.BpawnPositions, self.Wpawn, self.Bpawn)
        self.drawPieces(window, self.WrookPositions, self.BrookPositions, self.Wrook, self.Brook)
        self.drawPieces(window, self.WknightPositions, self.BknightPositions, self.Wknight, self.Bknight)
        self.drawPieces(window, self.WbishopPositions, self.BbishopPositions, self.Wbishop, self.Bbishop)
        self.drawPieces(window, self.WqueenPositions, self.BqueenPositions, self.Wqueen, self.Bqueen)
        self.drawPieces(window, self.WkingPositions, self.BkingPositions, self.Wking, self.Bking)


    def getBoardDict(self):

        boardDict = {}
        for posNotation in list(self.coordinateDict.keys()):
            boardDict[posNotation] = 0

        # Update Pawns

        for loc in self.WpawnPositions:
            boardDict[loc] = 1
        for loc in self.BpawnPositions:
            boardDict[loc] = -1

        for loc in self.WrookPositions:
            boardDict[loc] = 2
        for loc in self.BrookPositions:
            boardDict[loc] = -2

        for loc in self.WknightPositions:
            boardDict[loc] = 3
        for loc in self.BknightPositions:
            boardDict[loc] = -3

        for loc in self.WbishopPositions:
            boardDict[loc] = 4
        for loc in self.BbishopPositions:
            boardDict[loc] = -4

        for loc in self.WqueenPositions:
            boardDict[loc] = 5
        for loc in self.BqueenPositions:
            boardDict[loc] = -5

        for loc in self.WkingPositions:
            boardDict[loc] = 6
        for loc in self.BkingPositions:
            boardDict[loc] = -6

        return boardDict

    def drawPieces(self, window, posW, posB, imgW, imgB):

        # Draw white
        for posKey in posW:
            myCoords = self.coordinateDict[posKey]
            window.blit(imgW, (myCoords[0], myCoords[1]))

        for posKey in posB:
            myCoords = self.coordinateDict[posKey]
            window.blit(imgB, (myCoords[0], myCoords[1]))

    @staticmethod
    def CreateLocationDict():
        coordinateDict = {}
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for rowsNums in range(8):
            myRow = str(rowsNums + 1)
            myY = (8 - (rowsNums + 1)) * 100
            coordCounter = 0

            for columnLetters in letters:
                myColumn = columnLetters
                myKey = myColumn + myRow
                myX = coordCounter * 100
                myCoord = [myX, myY]

                coordCounter += 1

                coordinateDict[myKey] = myCoord

        return coordinateDict

    def getBoard(self, window):
        skip = False
        for rows in range(8):
            if skip:
                k = 1
            else:
                k = 0
            for i in range(0 + k, 8 + k, 2):
                pygame.draw.rect(window, (255, 178, 102),
                                 (i * self.sizeBetween, rows * self.sizeBetween, self.sizeBetween, self.sizeBetween))
                if k == 1:
                    skip = False
                elif k == 0:
                    skip = True

    def MouseMovement(self):
        mousePos = pygame.mouse.get_pos()
        mouseX = mousePos[0]
        mouseY = mousePos[1]
        mouseX_r = roundDown(mouseX)
        mouseY_r = roundDown(mouseY)

        notationCoord = get_key([mouseX_r, mouseY_r], self.coordinateDict)

        return mouseX, mouseY, notationCoord

    def CheckMouseTap(self, window):
        global quitGame
        mouseX, mouseY, notationCoord = self.MouseMovement()
        events = pygame.event.get()
        mypiece = 0
        pieceType = None
        mypiece_cord = None
        for event in events:

            if event.type == pygame.MOUSEBUTTONDOWN:

                self.selectedPos = notationCoord
                self.isSelected = False
                if self.boardDict[self.selectedPos] > 0 and self.whosTurn == 'W':  # Whites Turn
                    pieceType = self.valueNotationDict[self.boardDict[self.selectedPos]][1]
                    mypiece_cord = notationCoord
                    self.isSelected = True
                if self.boardDict[self.selectedPos] < 0 and self.whosTurn == 'B':  # Blacks Turn
                    pieceType = self.valueNotationDict[self.boardDict[self.selectedPos]][1]
                    mypiece_cord = notationCoord
                    self.isSelected = True
            if event.type == pygame.QUIT:
                quitGame = True
                quit()
        return pieceType, mypiece_cord

    def getPossibleStates(self, window):
        global quitGame
        boardlist = list(self.boardDict)
        mouseX, mouseY, notationCoord = self.MouseMovement()
        self.possibleStateW = {}
        self.possibleStateB = {}
        self.selectedPiece, selectedPieceCord = self.CheckMouseTap(window)
        if selectedPieceCord is not None:
            self.selectedPieceCoord = self.coordinateDict[selectedPieceCord]
            self.selectedPieceNotation = selectedPieceCord

        realPos = None
        for notationPos in boardlist:
            numOfPotStates = 0
            self.allPossibleForSelected[notationPos] = []
            self.boardDict = self.getBoardDict()

            # Check which type of piece
            try:
                typePiece = self.valueNotationDict[self.boardDict[notationPos]][1]
            except:
                typePiece = 'None'

            if typePiece == 'Wpawn' and self.whosTurn == 'W':  # If type white pawn

                # find which column the piece is

                realPos = self.coordinateDict[notationPos]
                piecevalue = 1

                # Assign possible positions

                pos_possibleFront1 = [realPos[0], realPos[1] - self.sizeBetween]
                try:
                    pos_notationFront1 = get_key(pos_possibleFront1, self.coordinateDict)
                except:
                    pos_notationFront1 = -1

                pos_possibleFront2 = [realPos[0], realPos[1] - 2 * self.sizeBetween]
                try:
                    pos_notationFront2 = get_key(pos_possibleFront2, self.coordinateDict)
                except:
                    pos_notationFront2 = -1

                pos_possibleAttackLeft = [realPos[0] - self.sizeBetween, realPos[1] - self.sizeBetween]
                try:
                    pos_notationAttackLeft = get_key(pos_possibleAttackLeft, self.coordinateDict)
                except:
                    pos_notationAttackLeft = -1

                pos_possibleAttackRight = [realPos[0] + self.sizeBetween, realPos[1] - self.sizeBetween]
                try:
                    pos_notationAttackRight = get_key(pos_possibleAttackRight, self.coordinateDict)
                except:
                    pos_notationAttackRight = -1

                # Check Possible Positions, add to possible state, add to possible for selected to draw

                # front1
                if pos_notationFront1 != -1:
                    if self.boardDict[pos_notationFront1] == 0:  # ### MAY ERROR
                        # Remove previous position from board dict
                        posState = copy(self.boardDict)
                        posState[notationPos] = 0
                        posState[pos_notationFront1] = piecevalue
                        self.allPossibleForSelected[notationPos].append(pos_notationFront1)
                        self.possibleStateW[
                            typePiece + '_' + notationPos + '_to_' + pos_notationFront1 + '_PosState'] = posState
                        # print(typePiece + '_' + notationPos + '_to_' + pos_notationFront1 + '_PosState')
                        # print(posState)

                # front2
                if pos_notationFront1 != -1 and pos_notationFront2 != -1:
                    if self.boardDict[pos_notationFront2] == 0 and self.boardDict[pos_notationFront1] == 0 and \
                            notationPos[
                                1] == '2':
                        posState = copy(self.boardDict)
                        posState[notationPos] = 0
                        posState[pos_notationFront2] = piecevalue
                        self.allPossibleForSelected[notationPos].append(pos_notationFront2)
                        self.possibleStateW[
                            typePiece + '_' + notationPos + '_to_' + pos_notationFront2 + '_PosState'] = posState

                if pos_notationAttackLeft != -1:
                    try:
                        if self.boardDict[pos_notationAttackLeft] < 0:
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[pos_notationAttackLeft] = piecevalue
                            self.allPossibleForSelected[notationPos].append(pos_notationAttackLeft)
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + pos_notationAttackLeft + '_PosState'] = posState
                    except:
                        pass

                if pos_notationAttackRight != -1:
                    try:
                        if self.boardDict[pos_notationAttackRight] < 0:
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[pos_notationAttackRight] = piecevalue
                            self.allPossibleForSelected[notationPos].append(pos_notationAttackRight)
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + pos_notationAttackRight + '_PosState'] = posState
                    except:
                        pass

            elif typePiece == 'Wrook' and self.whosTurn == 'W':
                realPos = self.coordinateDict[notationPos]
                piecevalue = 2

                # Assign possible positions:

                # Check y axis

                # check up, check distance to up border
                possiblePositionsUp = []
                myDistUp = realPos[1]
                howManySquares = int(myDistUp / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0],
                                   realPos[1] - a_square * self.sizeBetween]  # didnt check if it is possible yet!!!
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    # check is moveable?
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check down, check distance to down border
                possiblePositionsDown = []
                myDistDown = self.WINSIZE - realPos[1] - self.sizeBetween
                howManySquares = int(myDistDown / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0], realPos[1] + a_square * self.sizeBetween]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass
                # Check x axis
                # check left check distance to left border

                possiblePositionsLeft = []
                myDistLeft = realPos[0]
                howManySquares = int(myDistLeft / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] - a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check right, check distance to right border

                possiblePositionsRight = []
                myDistRight = self.WINSIZE - realPos[0] - self.sizeBetween
                howManySquares = int(myDistRight / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] + a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                allPossible = possiblePositionsRight + possiblePositionsLeft + possiblePositionsUp + possiblePositionsDown
                self.allPossibleForSelected[notationPos] = allPossible



            elif typePiece == 'Wbishop' and self.whosTurn == 'W':  # If type white bishop

                # find which coordinate the piece is
                realPos = self.coordinateDict[notationPos]
                self.blocked = False
                downright_blocked = False
                downleft_blocked = False
                upright_blocked = False
                upleft_blocked = False
                self.boardDict = self.getBoardDict()
                myState = self.getBoardDict()
                piecevalue = 4

                possiblepos_downright = []
                possiblepos_downleft = []
                possiblepos_upright = []
                possiblepos_upleft = []
                allPossible = []

                # For down right for (x,y) ---> (x+d, y+d) - d is the distance between each square
                # For down left for (x,y) ---> (x-d, y+d) - d is the distance between each square
                # For up right for (x,y) ---> (x+d, y-d) - d is the distance between each square
                # For up left for (x,y) ---> (x-d, y-d) - d is the distance between each square

                for dist in range(1, 8):
                    d = dist * self.sizeBetween
                    possible_position_downright = [realPos[0] + d, realPos[1] + d]
                    possible_position_downleft = [realPos[0] - d, realPos[1] + d]
                    possible_position_upright = [realPos[0] + d, realPos[1] - d]
                    possible_position_upleft = [realPos[0] - d, realPos[1] - d]

                    # DOWN-RIGHT

                    possible_notation_downright = get_key(possible_position_downright, self.coordinateDict)

                    if possible_notation_downright != -1:
                        if myState[possible_notation_downright] == 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                        elif myState[possible_notation_downright] < 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                            downright_blocked = True
                        elif myState[possible_notation_downright] > 0 and not downright_blocked:
                            downright_blocked = True
                        else:
                            pass

                    # DOWN-LEFT

                    possible_notation_downleft = get_key(possible_position_downleft, self.coordinateDict)

                    if possible_notation_downleft != -1:
                        if myState[possible_notation_downleft] == 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                        elif myState[possible_notation_downleft] < 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                            downleft_blocked = True
                        elif myState[possible_notation_downleft] > 0 and not downleft_blocked:
                            downleft_blocked = True
                        else:
                            pass

                    # UP-RIGHT

                    possible_notation_upright = get_key(possible_position_upright, self.coordinateDict)

                    if possible_notation_upright != -1:

                        if myState[possible_notation_upright] == 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState
                        elif myState[possible_notation_upright] < 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState

                            upright_blocked = True
                        elif myState[possible_notation_upright] > 0 and not upright_blocked:
                            upright_blocked = True
                        else:
                            pass

                    # UP-LEFT

                    possible_notation_upleft = get_key(possible_position_upleft, self.coordinateDict)

                    if possible_notation_upleft != -1:

                        if myState[possible_notation_upleft] == 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                        elif myState[possible_notation_upleft] < 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                            upleft_blocked = True
                        elif myState[possible_notation_upleft] > 0 and not upleft_blocked:
                            upleft_blocked = True
                        else:
                            pass


            elif typePiece == 'Wqueen' and self.whosTurn == 'W':  # If type white queen
                # find which coordinate the piece is
                realPos = self.coordinateDict[notationPos]
                self.blocked = False
                downright_blocked = False
                downleft_blocked = False
                upright_blocked = False
                upleft_blocked = False
                self.boardDict = self.getBoardDict()
                myState = self.getBoardDict()
                piecevalue = 5

                possiblepos_downright = []
                possiblepos_downleft = []
                possiblepos_upright = []
                possiblepos_upleft = []

                # For down right for (x,y) ---> (x+d, y+d) - d is the distance between each square
                # For down left for (x,y) ---> (x-d, y+d) - d is the distance between each square
                # For up right for (x,y) ---> (x+d, y-d) - d is the distance between each square
                # For up left for (x,y) ---> (x-d, y-d) - d is the distance between each square

                for dist in range(1, 8):
                    d = dist * self.sizeBetween
                    possible_position_downright = [realPos[0] + d, realPos[1] + d]
                    possible_position_downleft = [realPos[0] - d, realPos[1] + d]
                    possible_position_upright = [realPos[0] + d, realPos[1] - d]
                    possible_position_upleft = [realPos[0] - d, realPos[1] - d]

                    # DOWN-RIGHT

                    possible_notation_downright = get_key(possible_position_downright, self.coordinateDict)

                    if possible_notation_downright != -1:
                        if myState[possible_notation_downright] == 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                        elif myState[possible_notation_downright] < 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                            downright_blocked = True
                        elif myState[possible_notation_downright] > 0 and not downright_blocked:
                            downright_blocked = True
                        else:
                            pass

                    # DOWN-LEFT

                    possible_notation_downleft = get_key(possible_position_downleft, self.coordinateDict)

                    if possible_notation_downleft != -1:
                        if myState[possible_notation_downleft] == 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                        elif myState[possible_notation_downleft] < 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                            downleft_blocked = True
                        elif myState[possible_notation_downleft] > 0 and not downleft_blocked:
                            downleft_blocked = True
                        else:
                            pass

                    # UP-RIGHT

                    possible_notation_upright = get_key(possible_position_upright, self.coordinateDict)

                    if possible_notation_upright != -1:

                        if myState[possible_notation_upright] == 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState
                        elif myState[possible_notation_upright] < 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState

                            upright_blocked = True
                        elif myState[possible_notation_upright] > 0 and not upright_blocked:
                            upright_blocked = True
                        else:
                            pass

                    # UP-LEFT

                    possible_notation_upleft = get_key(possible_position_upleft, self.coordinateDict)

                    if possible_notation_upleft != -1:

                        if myState[possible_notation_upleft] == 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                        elif myState[possible_notation_upleft] < 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                            upleft_blocked = True
                        elif myState[possible_notation_upleft] > 0 and not upleft_blocked:
                            upleft_blocked = True
                        else:
                            pass

                allForDiagonal = possiblepos_downright + possiblepos_downleft + possiblepos_upleft + possiblepos_upright


                possiblePositionsUp = []
                myDistUp = realPos[1]
                howManySquares = int(myDistUp / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0],
                                   realPos[1] - a_square * self.sizeBetween]  # didnt check if it is possible yet!!!
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    # check is moveable?
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check down, check distance to down border
                possiblePositionsDown = []
                myDistDown = self.WINSIZE - realPos[1] - self.sizeBetween
                howManySquares = int(myDistDown / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0], realPos[1] + a_square * self.sizeBetween]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass
                # Check x axis
                # check left check distance to left border

                possiblePositionsLeft = []
                myDistLeft = realPos[0]
                howManySquares = int(myDistLeft / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] - a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check right, check distance to right border

                possiblePositionsRight = []
                myDistRight = self.WINSIZE - realPos[0] - self.sizeBetween
                howManySquares = int(myDistRight / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] + a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                allPossible = possiblePositionsRight + possiblePositionsLeft + possiblePositionsUp + possiblePositionsDown + allForDiagonal
                self.allPossibleForSelected[notationPos] = allPossible

            elif typePiece == 'Wknight' and self.whosTurn == 'W':  # If type white knight

                # find which column the piece is

                realPos = self.coordinateDict[notationPos]
                piecevalue = 3
                possiblePositions = []

                possible_x = [realPos[0]+self.sizeBetween, realPos[0]+2 * self.sizeBetween, realPos[0]+2 * self.sizeBetween,
                              realPos[0]+self.sizeBetween, realPos[0]-self.sizeBetween, realPos[0]-2 * self.sizeBetween,
                              realPos[0]-2 * self.sizeBetween, realPos[0]-self.sizeBetween]
                possible_y = [realPos[1] -2 * self.sizeBetween, realPos[1] - self.sizeBetween, realPos[1] + self.sizeBetween,
                              realPos[1] +2 * self.sizeBetween, realPos[1] +2 * self.sizeBetween, realPos[1]+ self.sizeBetween,
                              realPos[1]-self.sizeBetween, realPos[1]-2 * self.sizeBetween]

                for i in range(8): # Because knight has 8 possible moves
                    possible_coord = [possible_x[i], possible_y[i]]
                    possible_notation = get_key(possible_coord,self.coordinateDict)

                    if possible_notation != -1:
                        possible_notation_onboard = self.boardDict[possible_notation]

                        if possible_notation_onboard <= 0:
                            possiblePositions.append(possible_notation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation + '_PosState'] = posState

                self.allPossibleForSelected[notationPos] = possiblePositions

            elif typePiece == 'Wking' and self.whosTurn == 'W':  # If type white knight

                # find which column the piece is

                realPos = self.coordinateDict[notationPos]
                x = realPos[0]
                y = realPos[1]
                piecevalue = 6
                possiblePositions = []

                possible_x = [x, x+self.sizeBetween, x+self.sizeBetween, x+self.sizeBetween, x, x-self.sizeBetween,
                              x-self.sizeBetween, x-self.sizeBetween]
                possible_y = [y-self.sizeBetween, y-self.sizeBetween, y, y+self.sizeBetween, y+self.sizeBetween,
                              y+self.sizeBetween, y, y-self.sizeBetween]

                for i in range(8): # Because king has 8 possible moves
                    possible_coord = [possible_x[i], possible_y[i]]
                    possible_notation = get_key(possible_coord,self.coordinateDict)

                    if possible_notation != -1:
                        possible_notation_onboard = self.boardDict[possible_notation]

                        if possible_notation_onboard <= 0:
                            possiblePositions.append(possible_notation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation] = piecevalue
                            self.possibleStateW[
                                typePiece + '_' + notationPos + '_to_' + possible_notation + '_PosState'] = posState

                self.allPossibleForSelected[notationPos] = possiblePositions





            ######################################################################################

            if typePiece == 'Bpawn' and self.whosTurn == 'B':  # If type black pawn

                # find which column the piece is

                realPos = self.coordinateDict[notationPos]
                piecevalue = -1

                # Assign possible positions

                pos_possibleFront1 = [realPos[0], realPos[1] + self.sizeBetween]
                try:
                    pos_notationFront1 = get_key(pos_possibleFront1, self.coordinateDict)
                except:
                    pos_notationFront1 = -1

                pos_possibleFront2 = [realPos[0], realPos[1] + 2 * self.sizeBetween]
                try:
                    pos_notationFront2 = get_key(pos_possibleFront2, self.coordinateDict)
                except:
                    pos_notationFront2 = -1

                pos_possibleAttackLeft = [realPos[0] + self.sizeBetween, realPos[1] + self.sizeBetween]
                try:
                    pos_notationAttackLeft = get_key(pos_possibleAttackLeft, self.coordinateDict)
                except:
                    pos_notationAttackLeft = -1

                pos_possibleAttackRight = [realPos[0] - self.sizeBetween, realPos[1] + self.sizeBetween]
                try:
                    pos_notationAttackRight = get_key(pos_possibleAttackRight, self.coordinateDict)
                except:
                    pos_notationAttackRight = -1

                # Check Possible Positions, add to possible state, add to possible for selected to draw

                # front1
                if pos_notationFront1 != -1:
                    if self.boardDict[pos_notationFront1] == 0:  # ### MAY ERROR
                        # Remove previous position from board dict
                        posState = copy(self.boardDict)
                        posState[notationPos] = 0
                        posState[pos_notationFront1] = piecevalue
                        self.allPossibleForSelected[notationPos].append(pos_notationFront1)
                        self.possibleStateB[
                            typePiece + '_' + notationPos + '_to_' + pos_notationFront1 + '_PosState'] = posState
                        # print(typePiece + '_' + notationPos + '_to_' + pos_notationFront1 + '_PosState')
                        # print(posState)

                # front2
                if pos_notationFront1 != -1 and pos_notationFront2 != -1:
                    if self.boardDict[pos_notationFront2] == 0 and self.boardDict[pos_notationFront1] == 0 and \
                            notationPos[
                                1] == '7':
                        posState = copy(self.boardDict)
                        posState[notationPos] = 0
                        posState[pos_notationFront2] = piecevalue
                        self.allPossibleForSelected[notationPos].append(pos_notationFront2)
                        self.possibleStateB[
                            typePiece + '_' + notationPos + '_to_' + pos_notationFront2 + '_PosState'] = posState

                if pos_notationAttackLeft != -1:
                    try:
                        if self.boardDict[pos_notationAttackLeft] > 0:
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[pos_notationAttackLeft] = piecevalue
                            self.allPossibleForSelected[notationPos].append(pos_notationAttackLeft)
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + pos_notationAttackLeft + '_PosState'] = posState
                    except:
                        pass

                if pos_notationAttackRight != -1:
                    try:
                        if self.boardDict[pos_notationAttackRight] > 0:
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[pos_notationAttackRight] = piecevalue
                            self.allPossibleForSelected[notationPos].append(pos_notationAttackRight)
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + pos_notationAttackRight + '_PosState'] = posState
                    except:
                        pass

            elif typePiece == 'Brook' and self.whosTurn == 'B':
                realPos = self.coordinateDict[notationPos]
                piecevalue = -2

                # Assign possible positions:

                # Check y axis

                # check up, check distance to up border
                possiblePositionsUp = []
                myDistUp = realPos[1]
                howManySquares = int(myDistUp / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0],
                                   realPos[1] - a_square * self.sizeBetween]  # didnt check if it is possible yet!!!
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    # check is moveable?
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check down, check distance to down border
                possiblePositionsDown = []
                myDistDown = self.WINSIZE - realPos[1] - self.sizeBetween
                howManySquares = int(myDistDown / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0], realPos[1] + a_square * self.sizeBetween]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass
                # Check x axis
                # check left check distance to left border

                possiblePositionsLeft = []
                myDistLeft = realPos[0]
                howManySquares = int(myDistLeft / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] - a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check right, check distance to right border

                possiblePositionsRight = []
                myDistRight = self.WINSIZE - realPos[0] - self.sizeBetween
                howManySquares = int(myDistRight / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] + a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                allPossible = possiblePositionsRight + possiblePositionsLeft + possiblePositionsUp + possiblePositionsDown
                self.allPossibleForSelected[notationPos] = allPossible



            elif typePiece == 'Bbishop' and self.whosTurn == 'B':  # If type black pawn

                # find which coordinate the piece is
                realPos = self.coordinateDict[notationPos]
                self.blocked = False
                downright_blocked = False
                downleft_blocked = False
                upright_blocked = False
                upleft_blocked = False
                self.boardDict = self.getBoardDict()
                myState = self.getBoardDict()
                piecevalue = -4

                possiblepos_downright = []
                possiblepos_downleft = []
                possiblepos_upright = []
                possiblepos_upleft = []
                allPossible = []

                # For down right for (x,y) ---> (x+d, y+d) - d is the distance between each square
                # For down left for (x,y) ---> (x-d, y+d) - d is the distance between each square
                # For up right for (x,y) ---> (x+d, y-d) - d is the distance between each square
                # For up left for (x,y) ---> (x-d, y-d) - d is the distance between each square

                for dist in range(1, 8):
                    d = dist * self.sizeBetween
                    possible_position_downright = [realPos[0] + d, realPos[1] + d]
                    possible_position_downleft = [realPos[0] - d, realPos[1] + d]
                    possible_position_upright = [realPos[0] + d, realPos[1] - d]
                    possible_position_upleft = [realPos[0] - d, realPos[1] - d]

                    # DOWN-RIGHT

                    possible_notation_downright = get_key(possible_position_downright, self.coordinateDict)

                    if possible_notation_downright != -1:
                        if myState[possible_notation_downright] == 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                        elif myState[possible_notation_downright] > 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                            downright_blocked = True
                        elif myState[possible_notation_downright] < 0 and not downright_blocked:
                            downright_blocked = True
                        else:
                            pass

                    # DOWN-LEFT

                    possible_notation_downleft = get_key(possible_position_downleft, self.coordinateDict)

                    if possible_notation_downleft != -1:
                        if myState[possible_notation_downleft] == 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                        elif myState[possible_notation_downleft] > 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                            downleft_blocked = True
                        elif myState[possible_notation_downleft] < 0 and not downleft_blocked:
                            downleft_blocked = True
                        else:
                            pass

                    # UP-RIGHT

                    possible_notation_upright = get_key(possible_position_upright, self.coordinateDict)

                    if possible_notation_upright != -1:

                        if myState[possible_notation_upright] == 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState
                        elif myState[possible_notation_upright] > 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = self.boardDict[notationPos]
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState

                            upright_blocked = True
                        elif myState[possible_notation_upright] < 0 and not upright_blocked:
                            upright_blocked = True
                        else:
                            pass

                    # UP-LEFT

                    possible_notation_upleft = get_key(possible_position_upleft, self.coordinateDict)

                    if possible_notation_upleft != -1:

                        if myState[possible_notation_upleft] == 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                        elif myState[possible_notation_upleft] > 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                            upleft_blocked = True
                        elif myState[possible_notation_upleft] < 0 and not upleft_blocked:
                            upleft_blocked = True
                        else:
                            pass

            elif typePiece == 'Bqueen' and self.whosTurn == 'B':  # If type white queen
                # find which coordinate the piece is
                realPos = self.coordinateDict[notationPos]
                self.blocked = False
                downright_blocked = False
                downleft_blocked = False
                upright_blocked = False
                upleft_blocked = False
                self.boardDict = self.getBoardDict()
                myState = self.getBoardDict()
                piecevalue = -5

                possiblepos_downright = []
                possiblepos_downleft = []
                possiblepos_upright = []
                possiblepos_upleft = []

                # For down right for (x,y) ---> (x+d, y+d) - d is the distance between each square
                # For down left for (x,y) ---> (x-d, y+d) - d is the distance between each square
                # For up right for (x,y) ---> (x+d, y-d) - d is the distance between each square
                # For up left for (x,y) ---> (x-d, y-d) - d is the distance between each square

                for dist in range(1, 8):
                    d = dist * self.sizeBetween
                    possible_position_downright = [realPos[0] + d, realPos[1] + d]
                    possible_position_downleft = [realPos[0] - d, realPos[1] + d]
                    possible_position_upright = [realPos[0] + d, realPos[1] - d]
                    possible_position_upleft = [realPos[0] - d, realPos[1] - d]

                    # DOWN-RIGHT

                    possible_notation_downright = get_key(possible_position_downright, self.coordinateDict)

                    if possible_notation_downright != -1:
                        if myState[possible_notation_downright] == 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                        elif myState[possible_notation_downright] > 0 and not downright_blocked:
                            possiblepos_downright.append(possible_notation_downright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downright] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downright + '_PosState'] = posState
                            downright_blocked = True
                        elif myState[possible_notation_downright] < 0 and not downright_blocked:
                            downright_blocked = True
                        else:
                            pass

                    # DOWN-LEFT

                    possible_notation_downleft = get_key(possible_position_downleft, self.coordinateDict)

                    if possible_notation_downleft != -1:
                        if myState[possible_notation_downleft] == 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                        elif myState[possible_notation_downleft] > 0 and not downleft_blocked:
                            possiblepos_downleft.append(possible_notation_downleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_downleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_downleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_downleft + '_PosState'] = posState
                            downleft_blocked = True
                        elif myState[possible_notation_downleft] < 0 and not downleft_blocked:
                            downleft_blocked = True
                        else:
                            pass

                    # UP-RIGHT

                    possible_notation_upright = get_key(possible_position_upright, self.coordinateDict)

                    if possible_notation_upright != -1:

                        if myState[possible_notation_upright] == 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState
                        elif myState[possible_notation_upright] > 0 and not upright_blocked:
                            possiblepos_upright.append(possible_notation_upright)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upright)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upright] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upright + '_PosState'] = posState

                            upright_blocked = True
                        elif myState[possible_notation_upright] < 0 and not upright_blocked:
                            upright_blocked = True
                        else:
                            pass

                    # UP-LEFT

                    possible_notation_upleft = get_key(possible_position_upleft, self.coordinateDict)

                    if possible_notation_upleft != -1:

                        if myState[possible_notation_upleft] == 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                        elif myState[possible_notation_upleft] > 0 and not upleft_blocked:
                            possiblepos_upleft.append(possible_notation_upleft)
                            self.allPossibleForSelected[notationPos].append(possible_notation_upleft)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation_upleft] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation_upleft + '_PosState'] = posState
                            upleft_blocked = True
                        elif myState[possible_notation_upleft] < 0 and not upleft_blocked:
                            upleft_blocked = True
                        else:
                            pass

                allForDiagonal = possiblepos_downright + possiblepos_downleft + possiblepos_upleft + possiblepos_upright


                possiblePositionsUp = []
                myDistUp = realPos[1]
                howManySquares = int(myDistUp / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0],
                                   realPos[1] - a_square * self.sizeBetween]  # didnt check if it is possible yet!!!
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    # check is moveable?
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsUp.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check down, check distance to down border
                possiblePositionsDown = []
                myDistDown = self.WINSIZE - realPos[1] - self.sizeBetween
                howManySquares = int(myDistDown / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0], realPos[1] + a_square * self.sizeBetween]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsDown.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass
                # Check x axis
                # check left check distance to left border

                possiblePositionsLeft = []
                myDistLeft = realPos[0]
                howManySquares = int(myDistLeft / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] - a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsLeft.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                # check right, check distance to right border

                possiblePositionsRight = []
                myDistRight = self.WINSIZE - realPos[0] - self.sizeBetween
                howManySquares = int(myDistRight / self.sizeBetween)
                self.blocked = False

                for a_square in range(1, howManySquares + 1):
                    possiblePos = [realPos[0] + a_square * self.sizeBetween, realPos[1]]
                    possibleNotation = get_key(possiblePos, self.coordinateDict)
                    try:
                        if self.boardDict[possibleNotation] == 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                        elif self.boardDict[possibleNotation] > 0 and not self.blocked:
                            possiblePositionsRight.append(possibleNotation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possibleNotation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possibleNotation + '_PosState'] = posState
                            self.blocked = True
                        elif self.boardDict[possibleNotation] < 0 and not self.blocked:
                            self.blocked = True
                    except:
                        pass

                allPossible = possiblePositionsRight + possiblePositionsLeft + possiblePositionsUp + possiblePositionsDown + allForDiagonal
                self.allPossibleForSelected[notationPos] = allPossible

            elif typePiece == 'Bknight' and self.whosTurn == 'B':  # If type white knight

                # find which column the piece is

                realPos = self.coordinateDict[notationPos]
                piecevalue = -3
                possiblePositions = []

                possible_x = [realPos[0]+self.sizeBetween, realPos[0]+2 * self.sizeBetween, realPos[0]+2 * self.sizeBetween,
                              realPos[0]+self.sizeBetween, realPos[0]-self.sizeBetween, realPos[0]-2 * self.sizeBetween,
                              realPos[0]-2 * self.sizeBetween, realPos[0]-self.sizeBetween]
                possible_y = [realPos[1] -2 * self.sizeBetween, realPos[1] - self.sizeBetween, realPos[1] + self.sizeBetween,
                              realPos[1] +2 * self.sizeBetween, realPos[1] +2 * self.sizeBetween, realPos[1]+ self.sizeBetween,
                              realPos[1]-self.sizeBetween, realPos[1]-2 * self.sizeBetween]

                for i in range(8): # Because knight has 8 possible moves
                    possible_coord = [possible_x[i], possible_y[i]]
                    possible_notation = get_key(possible_coord,self.coordinateDict)

                    if possible_notation != -1:
                        possible_notation_onboard = self.boardDict[possible_notation]

                        if possible_notation_onboard >= 0:
                            possiblePositions.append(possible_notation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation + '_PosState'] = posState

                self.allPossibleForSelected[notationPos] = possiblePositions

            elif typePiece == 'Bking' and self.whosTurn == 'B':  # If type white knight

                # find which column the piece is

                realPos = self.coordinateDict[notationPos]
                x = realPos[0]
                y = realPos[1]
                piecevalue = -6
                possiblePositions = []

                possible_x = [x, x+self.sizeBetween, x+self.sizeBetween, x+self.sizeBetween, x, x-self.sizeBetween,
                              x-self.sizeBetween, x-self.sizeBetween]
                possible_y = [y-self.sizeBetween, y-self.sizeBetween, y, y+self.sizeBetween, y+self.sizeBetween,
                              y+self.sizeBetween, y, y-self.sizeBetween]

                for i in range(8): # Because king has 8 possible moves
                    possible_coord = [possible_x[i], possible_y[i]]
                    possible_notation = get_key(possible_coord,self.coordinateDict)

                    if possible_notation != -1:
                        possible_notation_onboard = self.boardDict[possible_notation]

                        if possible_notation_onboard >= 0:
                            possiblePositions.append(possible_notation)
                            posState = copy(self.boardDict)
                            posState[notationPos] = 0
                            posState[possible_notation] = piecevalue
                            self.possibleStateB[
                                typePiece + '_' + notationPos + '_to_' + possible_notation + '_PosState'] = posState

                self.allPossibleForSelected[notationPos] = possiblePositions

            #####################################################################
            if self.isSelected:

                pygame.draw.rect(window, (52, 253, 38),
                                 pygame.Rect(self.selectedPieceCoord[0], self.selectedPieceCoord[1],
                                             self.sizeBetween, self.sizeBetween), int(self.sizeBetween / 20))

                # Draw possible and move

                if len(self.allPossibleForSelected[self.selectedPieceNotation]) > 0:
                    possibleNotations = self.allPossibleForSelected[self.selectedPieceNotation]

                    for possibleSquare in possibleNotations:
                        myPos = self.coordinateDict[possibleSquare]
                        pygame.draw.rect(window, (52, 253, 38),
                                         pygame.Rect(myPos[0], myPos[1],
                                                     self.sizeBetween, self.sizeBetween), int(self.sizeBetween / 20))
            try:
                isInMoves, selectedPos = self.isInSelected()
                if isInMoves:
                    typePiece = self.boardDict[self.selectedPieceNotation]
                    PieceGroup = self.valueNotationDict[typePiece]
                    selectedType = self.boardDict[self.selectedPos]
                    selectedGroup = 0
                    if selectedType != 0:
                        selectedGroup = self.valueNotationDict[selectedType]

                    if PieceGroup[1] == 'Wpawn':
                        self.WpawnPositions.remove(self.selectedPieceNotation)
                        self.WpawnPositions.append(selectedPos)
                        if selectedPos[1] == '8':
                            self.WpawnPositions.remove(selectedPos)
                            self.WqueenPositions.append(selectedPos)

                    elif PieceGroup[1] == 'Wrook':
                        self.WrookPositions.remove(self.selectedPieceNotation)
                        self.WrookPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Wbishop':
                        self.WbishopPositions.remove(self.selectedPieceNotation)
                        self.WbishopPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Wqueen':
                        self.WqueenPositions.remove(self.selectedPieceNotation)
                        self.WqueenPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Wknight':
                        self.WknightPositions.remove(self.selectedPieceNotation)
                        self.WknightPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Wking':
                        self.WkingPositions.remove(self.selectedPieceNotation)
                        self.WkingPositions.append(selectedPos)

                    elif PieceGroup[1] == 'Bpawn':
                        self.BpawnPositions.remove(self.selectedPieceNotation)
                        self.BpawnPositions.append(selectedPos)
                        if selectedPos[1] == '1':
                            self.BpawnPositions.remove(selectedPos)
                            self.BqueenPositions.append(selectedPos)

                    elif PieceGroup[1] == 'Brook':
                        self.BrookPositions.remove(self.selectedPieceNotation)
                        self.BrookPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Bbishop':
                        self.BbishopPositions.remove(self.selectedPieceNotation)
                        self.BbishopPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Bqueen':
                        self.BqueenPositions.remove(self.selectedPieceNotation)
                        self.BqueenPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Bknight':
                        self.BknightPositions.remove(self.selectedPieceNotation)
                        self.BknightPositions.append(selectedPos)
                    elif PieceGroup[1] == 'Bking':
                        self.BkingPositions.remove(self.selectedPieceNotation)
                        self.BkingPositions.append(selectedPos)

                    if selectedType < 0:
                        if selectedGroup[1] == 'Brook':
                            self.BrookPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Bpawn':
                            self.BpawnPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Bknight':
                            self.BknightPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Bbishop':
                            self.BbishopPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Bqueen':
                            self.BqueenPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Bking':
                            self.BkingPositions.remove(self.selectedPos)
                            self.reward = 1
                            quitGame = True


                    if selectedType > 0:
                        if selectedGroup[1] == 'Wrook':
                            self.WrookPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Wpawn':
                            self.WpawnPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Wknight':
                            self.WknightPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Wbishop':
                            self.WbishopPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Wqueen':
                            self.WqueenPositions.remove(self.selectedPos)
                        elif selectedGroup[1] == 'Wking':
                            self.WkingPositions.remove(self.selectedPos)
                            self.reward = -1
                            quitGame = True

                        # ADD OTHER PIECES

                    if self.whosTurn == 'W':
                        self.whosTurn = 'B'
                        self.possibleStateW = {}

                    elif self.whosTurn == 'B':
                        self.whosTurn = 'W'
                        self.possibleStateB = {}

                    self.selectedPos = None

            except:
                pass

            # try:
            #     print(self.allPossibleForSelected[self.selectedPieceNotation])
            # except:
            #     pass
        # print(len(list(self.possibleState.keys())))

    def isInSelected(self):
        possMovesList = self.allPossibleForSelected[self.selectedPieceNotation]
        isInMoves = self.selectedPos in possMovesList
        return isInMoves, self.selectedPos
    def pawn_to_queeen(self):
        for notation in self.WpawnPositions:
            if notation[1] == '8':
                self.WpawnPositions.remove(notation)
                self.WqueenPositions.append(notation)
        for notation in self.BpawnPositions:
            if notation[1] == '1':
                self.BpawnPositions.remove(notation)
                self.BqueenPositions.append(notation)



    def getWindow(self):

        window = pygame.display.set_mode((self.WINSIZE, self.WINSIZE))
        pygame.display.set_caption("KCA chess")
        window.fill((153, 76, 5))

        self.getBoard(window)
        self.boardDict = self.getBoardDict()
        self.getPieces(window)
        self.pawn_to_queeen()
        self.valueNotationDict = self.getValueNotation()
        self.getPossibleStates(window)
        pygame.display.update()


def reformat_dict(dict_states):
    dicts_of_states = []
    list_of_states = []

    for move_name in dict_states.keys():
        dicts_of_states.append(dict_states[move_name])

    for board_dict in dicts_of_states:
        board_list = []
        for notation in board_dict.keys():
            piece_value = board_dict[notation]
            board_list.append(piece_value)
        list_of_states.append(board_list)

    return list_of_states



yin = DQNAgent()
yang = DQNAgent()
moves = 0
W_win = 0
B_win = 0
agentGeneration = 1


def StartGame():
    global quitGame,moves, W_win, B_win, agentGeneration
    myGame = GameEnv()
    clock = pygame.time.Clock()
    done = False
    quitGame = False
    moves = 0

    reward_yin = 0 #  White
    reward_yang = 0 # Black
    while not quitGame:
        # time.sleep(0.5)
        myGame.getWindow()
        whos_turn = myGame.whosTurn
        states = myGame.get_output()
        list_of_states = reformat_dict(states)

        if whos_turn == 'W':
           value_list, preferredState, index = yin.evaulate(list_of_states, whos_turn)
          # print(preferredState)
           states_list = list(states)
           state_key_move_selected = states_list[index]
           state = states[state_key_move_selected]

        elif whos_turn == 'B':
           value_list, preferredState, index = yang.evaulate(list_of_states, whos_turn)
           states_list = list(states)
           state_key_move_selected = states_list[index]
           state = states[state_key_move_selected]
           print(value_list)


        # print(state_key_move_selected)
        myGame.board_to_position(state)

        moves += 1
        if moves >= 120:
            quitGame = True
            done = True
        myGame.getWindow()
        next_state = myGame.getBoardDict()

        if len(myGame.WkingPositions) == 0:
            reward_yin = -100
            reward_yang = 100
            quitGame = True
            done = True
            B_win += 1
        elif len(myGame.BkingPositions) == 0:
            reward_yin = 100
            reward_yang = -100
            quitGame = True
            done = True
            W_win += 1

        yin.remember(state,reward_yin,next_state, done)
        yang.remember(state,reward_yang,next_state, done)

        myGame.TurnOver()

        # clock.tick(60)
    # print(reward)

    if done:
        agentGeneration += 1
        print("Generation: {}".format(agentGeneration))
        yin.batch_size = moves
        yang.batch_size = moves
        yin.replay(yin.batch_size)
        yang.replay(yang.batch_size)
        yin.memory = deque(maxlen=120)
        yang.memory = deque(maxlen=120)
        #
        # yin.model.save_weights(r'saved/model_yin_{}.h5'.format(agentGeneration))
        # yang.model.save_weights(r'saved/model_yang_{}.h5'.format(agentGeneration))

total_games = 0

#
# yin.epsilon = 0.20
# yang.epsilon =0.20
#
# yin.model.load_weights(r'saved/model_yin_22801.h5')
# yang.model.load_weights(r'saved/model_yang_22801.h5')

NUM_OF_GAMES_PLAYED = 10000000
for i in range(NUM_OF_GAMES_PLAYED):
    StartGame()
    total_games += 1
    print('episode: {}/{}, e:{:.2}, moves : {}, W_win %{} , B_win %{}'.format(i+1, NUM_OF_GAMES_PLAYED, yin.epsilon, moves,
                                                                              (W_win/(total_games))*100, (B_win/(total_games))*100))
    if i%100 == 0:
        yin.model.save_weights(r'saved/model_yin_{}.h5'.format(agentGeneration))
        yang.model.save_weights(r'saved/model_yang_{}.h5'.format(agentGeneration))


