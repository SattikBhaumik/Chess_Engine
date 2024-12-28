#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "chess.h" // Include or implement a chess library

#define BOARD_SIZE 8

// Piece values and positional weights
const double PIECE_VALUES[] = {0.0, 1.0, 3.2, 3.3, 5.0, 9.5, 0.0};
const double PAWN_WEIGHTS[BOARD_SIZE][BOARD_SIZE] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
    {0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1},
    {0.05, 0.05, 0.1, 0.25, 0.25, 0.1, 0.05, 0.05},
    {0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0},
    {0.05, -0.05, -0.1, 0.0, 0.0, -0.1, -0.05, 0.05},
    {0.05, 0.1, 0.0, -0.2, -0.2, 0.0, 0.1, 0.05},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
};

// Function prototypes
double evaluate_board_topologically(Board *board);
void calculate_piece_clusters(Board *board, double *white_cluster, double *black_cluster);
double cluster_cohesion(int positions[][2], int count);
Move mcmc_move_selection(Board *board, int is_white, int iterations);
void play_magnus_carlsen_engine();

int main() {
    play_magnus_carlsen_engine();
    return 0;
}

double evaluate_board_topologically(Board *board) {
    double evaluation = 0.0;

    for (int square = 0; square < BOARD_SIZE * BOARD_SIZE; ++square) {
        Piece piece = get_piece_at(board, square);
        if (piece.type != NONE) {
            int row = square / BOARD_SIZE;
            int col = square % BOARD_SIZE;

            double value = PIECE_VALUES[piece.type];
            double weight = (piece.type == PAWN) ? PAWN_WEIGHTS[row][col] : 0.0;
            double sign = (piece.color == WHITE) ? 1.0 : -1.0;

            evaluation += sign * (value + weight);
        }
    }

    double white_cluster = 0.0, black_cluster = 0.0;
    calculate_piece_clusters(board, &white_cluster, &black_cluster);
    evaluation += 0.1 * (white_cluster - black_cluster);

    return evaluation;
}

void calculate_piece_clusters(Board *board, double *white_cluster, double *black_cluster) {
    int white_positions[BOARD_SIZE * BOARD_SIZE][2];
    int black_positions[BOARD_SIZE * BOARD_SIZE][2];
    int white_count = 0, black_count = 0;

    for (int square = 0; square < BOARD_SIZE * BOARD_SIZE; ++square) {
        Piece piece = get_piece_at(board, square);
        if (piece.type != NONE) {
            int row = square / BOARD_SIZE;
            int col = square % BOARD_SIZE;
            if (piece.color == WHITE) {
                white_positions[white_count][0] = row;
                white_positions[white_count][1] = col;
                white_count++;
            } else {
                black_positions[black_count][0] = row;
                black_positions[black_count][1] = col;
                black_count++;
            }
        }
    }

    *white_cluster = cluster_cohesion(white_positions, white_count);
    *black_cluster = cluster_cohesion(black_positions, black_count);
}

double cluster_cohesion(int positions[][2], int count) {
    if (count == 0) return 0.0;

    double centroid[2] = {0.0, 0.0};
    for (int i = 0; i < count; ++i) {
        centroid[0] += positions[i][0];
        centroid[1] += positions[i][1];
    }
    centroid[0] /= count;
    centroid[1] /= count;

    double cohesion = 0.0;
    for (int i = 0; i < count; ++i) {
        cohesion += sqrt(pow(positions[i][0] - centroid[0], 2) +
                         pow(positions[i][1] - centroid[1], 2));
    }

    return -cohesion;
}

Move mcmc_move_selection(Board *board, int is_white, int iterations) {
    Move legal_moves[MAX_MOVES];
    int move_count = get_legal_moves(board, legal_moves);

    if (move_count == 0) {
        return null_move();
    }

    double scores[MAX_MOVES];
    for (int i = 0; i < move_count; ++i) {
        make_move(board, legal_moves[i]);
        scores[i] = evaluate_board_topologically(board);
        undo_move(board);
    }

    double probabilities[MAX_MOVES];
    double total = 0.0;
    for (int i = 0; i < move_count; ++i) {
        probabilities[i] = exp(scores[i]);
        total += probabilities[i];
    }
    for (int i = 0; i < move_count; ++i) {
        probabilities[i] /= total;
    }

    double rand_value = (double)rand() / RAND_MAX;
    double cumulative = 0.0;
    for (int i = 0; i < move_count; ++i) {
        cumulative += probabilities[i];
        if (rand_value <= cumulative) {
            return legal_moves[i];
        }
    }

    return legal_moves[move_count - 1];
}

void play_magnus_carlsen_engine() {
    Board board;
    init_board(&board);

    int is_white_turn = 1;

    while (!is_game_over(&board)) {
        print_board(&board);

        Move move = mcmc_move_selection(&board, is_white_turn, 1000);
        if (!is_null_move(move)) {
            make_move(&board, move);
        } else {
            printf("%s has no legal moves. Game over!\n", is_white_turn ? "White" : "Black");
            break;
        }

        is_white_turn = !is_white_turn;
    }

    printf("Game result: %s\n", get_result(&board));
}
