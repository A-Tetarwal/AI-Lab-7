import random

# Tic-Tac-Toe Board States
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), 
                         (0, 3, 6), (1, 4, 7), (2, 5, 8), 
                         (0, 4, 8), (2, 4, 6)]
        for combo in win_conditions:
            if all(self.board[i] == letter for i in combo):
                return True
        return False

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def is_full(self):
        return ' ' not in self.board

    def print_board(self):
        for row in [self.board[i:i+3] for i in range(0, 9, 3)]:
            print('| ' + ' | '.join(row) + ' |')

# MENACE Implementation
class MENACE:
    def __init__(self):
        self.matchboxes = {}
        self.game_history = []

    def get_moves_for_state(self, board):
        state = ''.join(board)
        if state not in self.matchboxes:
            # Initialize weights only for available moves
            weights = []
            for i in range(9):
                if board[i] == ' ':
                    weights.append(1)
                else:
                    weights.append(0)
            self.matchboxes[state] = weights
        return self.matchboxes[state]

    def make_move(self, board):
        moves = self.get_moves_for_state(board)
        available_moves = [i for i, weight in enumerate(moves) if weight > 0 and board[i] == ' ']
        
        if not available_moves:
            return None
            
        weights = [moves[i] for i in available_moves]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice([i for i, spot in enumerate(board) if spot == ' '])
            
        move = random.choices(available_moves, weights=weights)[0]
        self.game_history.append((''.join(board), move))
        return move

    def update_weights(self, result):
        for state, move in self.game_history:
            if state not in self.matchboxes:
                continue
                
            moves = self.matchboxes[state]
            if result == 'win':
                moves[move] += 3
            elif result == 'loss':
                moves[move] = max(1, moves[move] - 1)
        self.game_history = []  # Clear history after updating

def play_game(menace, player_letter='X'):
    game = TicTacToe()
    turn = 'X'
    
    while True:
        if game.is_full():
            return 'draw', game.board
            
        if turn == player_letter:
            available = game.available_moves()
            if available:
                square = random.choice(available)
            else:
                return 'draw', game.board
        else:
            square = menace.make_move(game.board)
            if square is None:
                return 'draw', game.board
                
        game.make_move(square, turn)
        
        if game.current_winner:
            return turn, game.board
            
        turn = 'O' if turn == 'X' else 'X'

# Train MENACE
menace = MENACE()
wins = {'X': 0, 'O': 0, 'draw': 0}

print("Training MENACE...")
for i in range(10000):
    result, final_board = play_game(menace)
    wins[result if result != 'O' and result != 'X' else result] += 1
    
    if result == 'O':  # MENACE wins
        menace.update_weights('win')
    elif result == 'X':  # MENACE loses
        menace.update_weights('loss')
    else:  # Draw
        menace.update_weights('draw')
        
    if (i + 1) % 1000 == 0:
        print(f"Completed {i + 1} training games")
        print(f"Wins: {wins['O']}, Losses: {wins['X']}, Draws: {wins['draw']}")

print("\nTraining complete!")
print(f"Final statistics:")
print(f"MENACE wins: {wins['O']}")
print(f"MENACE losses: {wins['X']}")
print(f"Draws: {wins['draw']}")

