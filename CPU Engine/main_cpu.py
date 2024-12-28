import copy

def initialize_board():
    """Initialize the chessboard with pieces in their starting positions."""
    board = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    ]
    return board

def print_board(board):
    """Print the chessboard in a human-readable format."""
    for row in board:
        print(" ".join(row))
    print()

def is_in_bounds(x, y):
    """Check if a position is within the board boundaries."""
    return 0 <= x < 8 and 0 <= y < 8

def generate_moves(board, is_white):
    """Generate all possible moves for the current player."""
    moves = []
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if (is_white and piece.isupper()) or (not is_white and piece.islower()):
                moves.extend(get_piece_moves(board, x, y, piece))
    return moves

def get_piece_moves(board, x, y, piece):
    """Generate moves for a specific piece."""
    moves = []
    directions = {
        'P': [(-1, 0), (-1, -1), (-1, 1)],
        'p': [(1, 0), (1, -1), (1, 1)],
        'R': [(-1, 0), (1, 0), (0, -1), (0, 1)],
        'r': [(-1, 0), (1, 0), (0, -1), (0, 1)],
        'N': [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (2, -1), (2, 1), (1, -2), (1, 2)],
        'n': [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (2, -1), (2, 1), (1, -2), (1, 2)],
        'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        'b': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        'Q': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
        'q': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
        'K': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
        'k': [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    }
    if piece.upper() in directions:
        for dx, dy in directions[piece]:
            nx, ny = x + dx, y + dy
            while is_in_bounds(nx, ny):
                if board[nx][ny] == '.' or board[nx][ny].islower() != piece.islower():
                    moves.append(((x, y), (nx, ny)))
                if board[nx][ny] != '.':
                    break
                if piece.upper() in ['N', 'K', 'P']:
                    break
                nx += dx
                ny += dy
    return moves

def evaluate_board(board):
    """Evaluate the board and return a score."""
    piece_values = {
        'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
        'P': -1, 'N': -3, 'B': -3, 'R': -5, 'Q': -9, 'K': 0
    }
    score = 0
    for row in board:
        for piece in row:
            score += piece_values.get(piece, 0)
    return score

def minimax(board, depth, is_white, alpha, beta):
    """Minimax algorithm with alpha-beta pruning."""
    if depth == 0:
        return evaluate_board(board), None

    best_move = None
    if is_white:
        max_eval = float('-inf')
        for move in generate_moves(board, is_white):
            new_board = copy.deepcopy(board)
            make_move(new_board, move)
            evaluation, _ = minimax(new_board, depth - 1, not is_white, alpha, beta)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in generate_moves(board, is_white):
            new_board = copy.deepcopy(board)
            make_move(new_board, move)
            evaluation, _ = minimax(new_board, depth - 1, not is_white, alpha, beta)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval, best_move

def make_move(board, move):
    """Execute a move on the board."""
    (x1, y1), (x2, y2) = move
    board[x2][y2] = board[x1][y1]
    board[x1][y1] = '.'

def play_game():
    """Simulate a chess game."""
    board = initialize_board()
    is_white_turn = True
    for turn in range(50):
        print(f"Turn {turn + 1} ({'White' if is_white_turn else 'Black'}):")
        print_board(board)
        _, best_move = minimax(board, 3, is_white_turn, float('-inf'), float('inf'))
        if best_move:
            make_move(board, best_move)
        else:
            print(f"{'White' if is_white_turn else 'Black'} has no legal moves. Game over!")
            break
        is_white_turn = not is_white_turn

if __name__ == "__main__":
    play_game()
