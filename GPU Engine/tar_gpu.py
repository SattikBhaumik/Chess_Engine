import chess
import chess.engine
import torch
import copy

# Piece values for evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# Convert the board to a tensor representation
def board_to_tensor(board):
    """Converts a chess board into a tensor for GPU computations."""
    tensor = torch.zeros((8, 8), device='cuda')
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)
            row, col = divmod(square, 8)
            tensor[row, col] = value
    return tensor

# Evaluate the board using a tensor
def evaluate_board(board):
    """Evaluate the board position using GPU acceleration."""
    tensor = board_to_tensor(board)
    return torch.sum(tensor).item()

# Minimax algorithm with alpha-beta pruning

def minimax(board, depth, alpha, beta, is_white):
    """Minimax with alpha-beta pruning on GPU."""
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    legal_moves = list(board.legal_moves)
    best_move = None

    if is_white:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, not is_white)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, not is_white)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

# Play the game
def play_game():
    """Simulates a chess game between two GPU-accelerated engines."""
    board = chess.Board()
    is_white_turn = True

    while not board.is_game_over():
        print(board)
        print()
        _, best_move = minimax(board, depth=3, alpha=float('-inf'), beta=float('inf'), is_white=is_white_turn)
        if best_move:
            board.push(best_move)
        else:
            print("No legal moves available. Game over!")
            break
        is_white_turn = not is_white_turn

    print("Game result:", board.result())

if __name__ == "__main__":
    play_game()
