"""Preprocess EPD file before running tune.py."""
import chess
import numpy as np
import pickle
import sys

PROCESSED_FILE = 'processed.pickle'

PIECE_INDEX = {
    'K': 0,
    'Q': 1,
    'R': 2,
    'B': 3,
    'N': 4,
    'P': 5,
    'k': 6,
    'q': 7,
    'r': 8,
    'b': 9,
    'n': 10,
    'p': 11
}


def reformat_board(board):
    arr = np.zeros((12, 64))
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            symbol = piece.symbol()
            if symbol.isupper():
                index = i ^ 56
            else:
                index = i
            pindx = PIECE_INDEX[symbol]
            arr[pindx, index] = 1
    return arr


def yield_epds(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()


def board_and_outcome(epd):
    board, parsed_ops = chess.Board.from_epd(epd)
    res = parsed_ops['c0']
    if res == '1-0':
        outcome = 1.0
    elif res == '0-1':
        outcome = 0.0
    elif res == '1/2-1/2':
        outcome = 0.5
    else:
        raise ValueError(f'Cannot process {res}')
    return board, outcome


def preprocess(filename):
    print(f'Preprocessing {filename}...')
    boards = []
    outcomes = []
    for i, epd in enumerate(yield_epds(filename)):
        board, outcome = board_and_outcome(epd)
        board = reformat_board(board)
        boards.append(board)
        outcomes.append(outcome)
        if i % 1000 == 0:
            print(f'Processed epds: {i}\r', end="")
    print(f'Processed epds: {i}')
    print('Done')
    print(f'Dumping processed data to {PROCESSED_FILE}...')
    with open(PROCESSED_FILE, 'wb') as f:
        pickle.dump(boards, f)
        pickle.dump(outcomes, f)
    print('Done')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:\npython3 preprocess.py <epd_file>\n', file=sys.stderr)
        sys.exit(-1)
    preprocess(sys.argv[1])
