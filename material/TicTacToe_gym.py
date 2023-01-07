# 코드 출처: http://inventwithpython.com/chapter10.html
# 코드 출처: http://github.com/haje01/gym-tictactoe/blob/master/gym_tictactoe/env.py
# -> 그대로 사용하지는 않고 gym과 같이 변형해서 사용

# Tic Tac Toe
import gym
from gym import spaces
import os
import random
def drawBoard(board):
    # This function prints out the board that it was passed.
    # "board" is a list of 10 strings representing the board (ignore index 0)
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('   |   |')

def inputPlayerLetter():
    # Lets the player type which letter they want to be.
    # Returns a list with the player’s letter as the first item, and the computer's letter as the second.
    letter = ''
    while not (letter == 'X' or letter == 'O'):
        print('Do you want to be X or O?')
        letter = input().upper()

    # the first element in the list is the player’s letter, the second is the computer's letter.
    if letter == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']

def whoGoesFirst():
    # Randomly choose the player who goes first.
    if random.randint(0, 1) == 0:
        return 'computer'
    else:
        return 'player'
    
def playAgain():
    # This function returns True if the player wants to play again, otherwise it returns False.
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')

def makeMove(board, letter, move):
    board[move] = letter
    return board

def isWinner(bo, le):
    # Given a board and a player’s letter, this function returns True if that player has won.
    # We use bo instead of board and le instead of letter so we don’t have to type as much.
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or # across the top
    (bo[4] == le and bo[5] == le and bo[6] == le) or # across the middle
    (bo[1] == le and bo[2] == le and bo[3] == le) or # across the bottom
    (bo[7] == le and bo[4] == le and bo[1] == le) or # down the left side
    (bo[8] == le and bo[5] == le and bo[2] == le) or # down the middle
    (bo[9] == le and bo[6] == le and bo[3] == le) or # down the right side
    (bo[7] == le and bo[5] == le and bo[3] == le) or # diagonal
    (bo[9] == le and bo[5] == le and bo[1] == le)) # diagonal

def getBoardCopy(board):
    # Make a duplicate of the board list and return it the duplicate.
    dupeBoard = []
    for i in board:
        dupeBoard.append(i)
    return dupeBoard

def isSpaceFree(board, move):
    # Return true if the passed move is free on the passed board.
    return board[move] == ' '

def getPlayerMove(board):
    # Let the player type in their move.
    move = ' '
    while move not in '1 2 3 4 5 6 7 8 9'.split() or not isSpaceFree(board, int(move)):
        print('What is your next move? (1-9)')
        move = input()
    return int(move)

def chooseRandomMoveFromList(board, movesList):
    # Returns a valid move from the passed list on the passed board.
    # Returns None if there is no valid move.
    possibleMoves = []
    for i in movesList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)

    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None

def getComputerMove(board, computerLetter):
    # Given a board and the computer's letter, determine where to move and return that move.
    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'

    # Here is our algorithm for our Tic Tac Toe AI:
    # First, check if we can win in the next move
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, computerLetter, i)
            if isWinner(copy, computerLetter):
                return i

    # Check if the player could win on their next move, and block them.
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i

    # Try to take one of the corners, if they are free.
    move = chooseRandomMoveFromList(board, [1, 3, 7, 9])
    if move != None:
        return move

    # Try to take the center, if it is free.
    if isSpaceFree(board, 5):
        return 5
    # Move on one of the sides.
    return chooseRandomMoveFromList(board, [2, 4, 6, 8])

def isBoardFull(board):
    # Return True if every space on the board has been taken. Otherwise return False.
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True



class TicTacToeEnv(gym.Env):
    reward = 0
    is_done = False
    Observation = [' ']*10
    info = {
        'Round':0,
        'Player':'X',
        'Winner':None
    }
    action_space = spaces.Discrete(9)
    observation_space = spaces.Discrete(9)
    difficulty='easy'
    #def __init__(self):
    #def env(self):
    #    return self
        
    def reset(self):
        self.reward = 0
        #self.
        self.Observation = [' ']*10
        self.info['Round'] = 0
        self.info['Player'] = 'X'
        self.info['Winner'] = None
        self.is_done=False
        return self.Observation
    
    def render(self,mode='human',close=False):
        if close:
            return 
        drawBoard(self.Observation)

    def step(self,action):
        self.action_space = self.valid_action()
        assert action in self.action_space
        #if action not in self.action_space:
        #    action = random.choice(self.action_space)

        if self.is_done:
            #raise ValueError("Environment finished! Use .reset() option if you want to resume game")
            return self.Observation, 0, self.is_done, self.info
        # First agent move and victory check
        #possible_actions = self.valid_action()
        #if action not in possible_actions:
        #    raise ValueError('Your action is not valid!')
        #    return self.Observation, self.reward, self.is_done, self.info
        self.info['Round'] += 1
        self.info['Player'] = 'X'
        #self.NextObservation = makeMove(self.Observation,'X',action)
        self.Observation = makeMove(self.Observation,'X',action)
        #if isWinner(self.NextObservation,'x'):
        if isWinner(self.Observation,'x'):
            self.reward = 1
            self.info['Winner'] = 'X'
            self.is_done=True
        else:
            #if isBoardFull(self.NextObservation):
            if isBoardFull(self.Observation):
                self.reward = 0
                self.info['Winner'] = None
                self.is_done=True
            else:
                # Second agent(computer) move and victory check
                self.info['Round'] += 1
                self.info['Player'] = 'O'
                #self.Observation = self.NextObservation
                if self.difficulty != 'easy':
                    #move = getComputerMove(self.NextObservation, 'O') # code base moving
                    move = getComputerMove(self.Observation, 'O') # code base moving
                else:
                    self.action_space = self.valid_action()
                    move = random.choice(self.action_space) 
                self.NextObservation = makeMove(self.Observation,'O',move)
                if isWinner(self.NextObservation,'O'):
                    self.reward = -1
                    self.info['Winner'] = 'O'
                    self.is_done=True
                else:
                    if isBoardFull(self.NextObservation):
                        self.reward = 0
                        self.info['Winner'] = None
                        self.is_done=True
        self.Observation = self.NextObservation
        return self.Observation, self.reward, self.is_done, self.info
                
    def valid_action(self):
        # Let agent Knows which action is valid in current tic-tac-toe observation
        possibleMoves = []
        for i in range(1,10):
            if isSpaceFree(self.Observation, i):
                possibleMoves.append(i)
        return possibleMoves
