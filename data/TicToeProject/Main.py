import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #formula:  Z = XW + b
        self.output = np.dot(inputs, self.weights) + self.biases



class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# TIC-TAC-TOE FUNCTIONS
def print_board(board):
    #dictionary that converts numbers -> symbols.
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("+---+---+---+")
    for i in range(3):
        row = board[i*3:(i+1)*3]
        print(f"| {symbols[row[0]]} | {symbols[row[1]]} | {symbols[row[2]]} |")
        print("+---+---+---+")

def is_valid_move(board, move):
    return 0 <= move < 9 and board[move] == 0

def get_winner(board):
    win_conditions = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]
    for a, b, c in win_conditions:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0  # no winner

#  NEURAL NETWORK MODEL
class TicTacToeModel:
    def __init__(self):
        # Architecture: 9 inputs → 32 hidden neurons (ReLU) → 9 outputs (Softmax)
        self.layer1 = Layer_Dense(9, 32)
        self.activation1 = Activation_ReLU()
        self.layer2 = Layer_Dense(32, 9)
        self.activation2 = Activation_Softmax()

    def predict(self, board):
        # board must be a (1, 9) shaped array
        inputs = np.array(board, ndmin=2)
        self.layer1.forward(inputs) #output is (1 x 32)
        self.activation1.forward(self.layer1.output) #ReLu
        self.layer2.forward(self.activation1.output) #output (1 × 9)
        self.activation2.forward(self.layer2.output) #Softmax sum = 1
        return self.activation2.output[0]  # return probabilities for 9 moves

#  MAIN GAME
model = TicTacToeModel()

print("=== TIC-TAC-TOE NEURAL NETWORK (X = AI) ===")

board = [0] * 9
current_player = 1  # 1 = AI (X), -1 = Me (O)

while True:
    print_board(board)
    winner = get_winner(board)
    if winner != 0:
        print("Game Over! Winner:", "(AI)" if winner == 1 else "(You!!!)")
        break
    if 0 not in board:
        print("Draw!")
        break

    if current_player == 1:  # AI turn
        print("\n🤖 AI is thinking...")
        probs = model.predict(board)
        # Choose the move with the highest probability among empty cells
        for move in np.argsort(probs)[::-1]:  # best to worst
            if is_valid_move(board, move):
                board[move] = 1
                print(f"AI plays at position {move}")
                break
    else:  # My turn
        while True:
            try:
                move = int(input("\nYour move (0-8): "))
                if is_valid_move(board, move):
                    board[move] = -1
                    break
                else:
                    print("Invalid move! Cell already taken.")
            except:
                print("Enter a number 0-8")

    current_player *= -1