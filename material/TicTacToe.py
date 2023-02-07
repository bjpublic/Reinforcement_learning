# 출처 : https://github.com/maksimKorzh/tictactoe-mtcs/blob/master/src/tictactoe/tictactoe.py

# 
# AI that learns to play Tic Tac Toe usint Reinforcement Learning
# MCTS + NN
# 
# packages
import copy

# Tic Tac Toe board class
class Board():
    # create constructor (init board class instance)
    def __init__(self, board=None):
        self.player_1 = 'x'
        self.player_2 = 'o'
        self.empty_Square='.'

        # define board position
        self.position = {}

        # init (reset) board
        self.init_board()

        # create a copy of a previous board state if available
        if board is not None:
            self.__dict__ = copy.deepcopy(board.__dict__)

    # Init (reset) board
    def init_board(self):
        # loop over board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                # sef every board square to empty square
                self.position[row,col]=self.empty_Square

    # make move
    def make_move(self, row, col):
        board = Board(self)
        board.position[row,col] = self.player_1
        # swap players
        (board.player_1, board.player_2) = (board.player_2,board.player_1)
        
        # return new board state
        return board

    # get whether the game is drawn
    def is_draw(self):
        # loop over board squares
        for row,col in self.position:
            # empty square is available
            if self.position[row,col]==self.empty_Square:
                # this is not a draw
                return False
        # by default we return a draw
        return True

    # get whether the game is drawn
    def is_win(self):
        #############################
        # vertical sequence detection
        #############################
        
        #loop over board columns
        for col in range(3):
            # define winning sequence list
            winning_sequence = []

            # loop over board rows:
            for row in range(3):
                # if found same next element in the row
                if self.position[row, col] == self.player_2:
                    # update winning sequence
                    winning_sequence.append((row, col))
                # if we have 3 elemnts in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

        #############################
        # horizontal sequence detection
        #############################
        
        #loop over board columns
        for row in range(3):
            # define winning sequence list
            winning_sequence = []

            # loop over board rows:
            for col in range(3):
                # if found same next element in the row
                if self.position[row, col] == self.player_2:
                    # update winning sequence
                    winning_sequence.append((row, col))
                # if we have 3 elemnts in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

        #############################
        # 1st diagonal sequence detection
        #############################
        # define winning sequence list
        winning_sequence = []
        # loop over board rows:
        for row in range(3):
            # init column
            col = row
            # if found same next element in the row
            if self.position[row, col] == self.player_2:
                # update winning sequence
                winning_sequence.append((row, col))
            # if we have 3 elemnts in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True


        #############################
        # 2nd diagonal sequence detection
        #############################
        winning_sequence = []
        # loop over board rows:
        for row in range(3):
            # init column
            col = 3- row -1
            # if found same next element in the row
            if self.position[row, col] == self.player_2:
                # update winning sequence
                winning_sequence.append((row, col))
            # if we have 3 elemnts in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True

        # by default return non winning state
        return False

    # generate legal moves to play in the current position
    def generate_states(self):
        # define states list(move list - list of available actions to consider)
        actions = []

        # loop over  board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                # make sure that current square is empty
                if self.position[row, col] == self.empty_Square:
                    # append available action/board state to action list
                    actions.append((self.make_move(row, col)))

        # return the list of available actions (board class instances)
        return actions

    # main game_loop
    def game_loop(self, mcts=None):
        print('  \nTic Tac Toe 게임을 시작합니다.')
        print('   순서는 1,2처럼 타이핑하세요. 1은 "행", 2는 "열"을 의미합니다.')
        print('   "exit"은 게임을 종료합니다.')

        # print board
        print(self)

        # create MCTS instance

        # game loop
        while True:
            # user input
            user_input = input('>> ')
            # escape condition
            if str(user_input) == 'exit': break

            # skip empty input
            if user_input == '': continue

            try:
                # parse user input (move format [row, col]: 1,2)
                row = int(user_input.split(',')[0])-1
                col = int(user_input.split(',')[1])-1
                
                # check move legality
                if self.position[row, col] != self.empty_Square:
                    print(' 실행할 수 없는 행동')
                    continue
                
                # make move on board
                self = self.make_move(row,col)

                # make AI move here...
                if mcts is not None:
                    best_move = mcts.search(self)
                    
                    # legal moves available
                    try:
                        self = best_move.board
                    except:
                        pass

                # print board
                print(self)

                # check the game state
                if self.is_win():
                    print(f' 게임의 승자: "{self.player_2}"!')
                    break
                # check if the game is drawn
                elif self.is_draw():
                    print(f'무승부 게임!')
                    break

            except Exception as e:
                print(' Error:', e)
                print(' 실행할 수 없는 행동')
                print('   순서는 1,2처럼 타이핑하세요. 1은 "행", 2는 "열"을 의미합니다.')



    # print board state
    def __str__(self):
        # define board string representation
        board_string = ''
        # loop over board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                board_string+=f' {self.position[row,col]} '
            board_string += '\n'

        # prepend side to move
        if self.player_1 == 'x':
            board_string='\n---------------\n "x" 의 차례: \n---------------\n\n'+board_string
        elif self.player_1 == 'o':
            board_string='\n---------------\n "o" 의 차례: \n---------------\n\n'+board_string
        # return board string
        return board_string

if __name__ == '__main__':
    board = Board()
    # start game loop
    board.game_loop()