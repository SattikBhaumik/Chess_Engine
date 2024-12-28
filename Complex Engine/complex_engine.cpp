#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>
#include <algorithm>
#include "chess.hpp" // Include a chess library, e.g., ChessLib

using namespace std;

// Piece values and positional weights
const map<PieceType, double> PIECE_VALUES = {
    {PAWN, 1.0},
    {KNIGHT, 3.2},
    {BISHOP, 3.3},
    {ROOK, 5.0},
    {QUEEN, 9.5},
    {KING, 0.0} // King has infinite value, but we ignore it for now.
};

const vector<vector<double>> PAWN_WEIGHTS = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
    {0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1},
    {0.05, 0.05, 0.1, 0.25, 0.25, 0.1, 0.05, 0.05},
    {0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0},
    {0.05, -0.05, -0.1, 0.0, 0.0, -0.1, -0.05, 0.05},
    {0.05, 0.1, 0.0, -0.2, -0.2, 0.0, 0.1, 0.05},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
};

double evaluateBoardTopologically(const Board &board) {
    double evaluation = 0.0;

    for (const auto &square : ALL_SQUARES) {
        Piece piece = board.pieceAt(square);
        if (piece.type != NONE) {
            int row = square / 8;
            int col = square % 8;

            double value = PIECE_VALUES.at(piece.type);
            double weight = (piece.type == PAWN) ? PAWN_WEIGHTS[row][col] : 0.0;
            double sign = (piece.color == WHITE) ? 1.0 : -1.0;

            evaluation += sign * (value + weight);
        }
    }

    auto [whiteCluster, blackCluster] = calculatePieceClusters(board);
    evaluation += 0.1 * (whiteCluster - blackCluster);

    return evaluation;
}

tuple<double, double> calculatePieceClusters(const Board &board) {
    vector<pair<int, int>> whitePositions;
    vector<pair<int, int>> blackPositions;

    for (const auto &square : ALL_SQUARES) {
        Piece piece = board.pieceAt(square);
        if (piece.type != NONE) {
            int row = square / 8;
            int col = square % 8;
            if (piece.color == WHITE) {
                whitePositions.emplace_back(row, col);
            } else {
                blackPositions.emplace_back(row, col);
            }
        }
    }

    double whiteCluster = clusterCohesion(whitePositions);
    double blackCluster = clusterCohesion(blackPositions);

    return {whiteCluster, blackCluster};
}

double clusterCohesion(const vector<pair<int, int>> &positions) {
    if (positions.empty()) return 0.0;

    pair<double, double> centroid = {0.0, 0.0};
    for (const auto &pos : positions) {
        centroid.first += pos.first;
        centroid.second += pos.second;
    }
    centroid.first /= positions.size();
    centroid.second /= positions.size();

    double cohesion = 0.0;
    for (const auto &pos : positions) {
        cohesion += sqrt(pow(pos.first - centroid.first, 2) + pow(pos.second - centroid.second, 2));
    }

    return -cohesion; // Smaller cohesion means better clustering
}

Move mcmcMoveSelection(Board &board, bool isWhite, int iterations = 1000) {
    vector<Move> legalMoves = board.legalMoves();
    if (legalMoves.empty()) return Move();

    vector<double> scores;
    for (const auto &move : legalMoves) {
        board.makeMove(move);
        scores.push_back(evaluateBoardTopologically(board));
        board.undoMove();
    }

    vector<double> probabilities;
    double sumExp = 0.0;
    for (double score : scores) {
        double expScore = exp(score);
        probabilities.push_back(expScore);
        sumExp += expScore;
    }
    for (double &prob : probabilities) {
        prob /= sumExp;
    }

    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    return legalMoves[dist(gen)];
}

void playMagnusCarlsenEngine() {
    Board board;
    bool isWhiteTurn = true;

    while (!board.isGameOver()) {
        cout << board << endl;

        Move move = mcmcMoveSelection(board, isWhiteTurn);
        if (!move.isNull()) {
            board.makeMove(move);
        } else {
            cout << (isWhiteTurn ? "White" : "Black") << " has no legal moves. Game over!" << endl;
            break;
        }

        isWhiteTurn = !isWhiteTurn;
    }

    cout << "Game result: " << board.result() << endl;
}

int main() {
    playMagnusCarlsenEngine();
    return 0;
}
