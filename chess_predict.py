# -*- coding: utf-8 -*-

import torch
import chess
import chess.pgn
import io
import numpy as np
  
def chess_predict(pgn):

  # Model Parameters
  move_hist = 8
  # Neural net grouping options
  # "no_group"
  # "no_group_b2b_conv2d"
  # "group_by_move_hist"
  # "group_by_piece"
  group = "no_group"

  # Dual-Use for training and inference
  # Read a PGN sequence
  # Convert each move to to a board state (each piece has a layer)
  # Sequence through all the moves of the PGN
  def pgn_to_states(pgn):

    # Initialize
    pgn = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn)
    pgn.close()
    board = game.board()
    board_states = []
    board_fen = []
    
    piece_to_num = {
      'P': 0,  # Pawn
      'R': 1,  # Rook
      'N': 2,  # Knight
      'B': 3,  # Bishop
      'Q': 4,  # Queen
      'K': 5}  # King

    for move in game.mainline_moves():

      # There are 6 types of pieces on an 8x8 board
      # Initialize board to all zeroes then add pieces per layer
      # as either +1 or -1
      board_state = np.zeros((6, 8, 8), dtype=np.int8)
      board.push(move)
      board_fen.append((board.fen(en_passant='fen')))
      
      for row in range(8):
        for col in range(8):
          piece_type = board.piece_at(chess.square(row, col))
          if piece_type:
            piece = str(piece_type)
            color = int(piece.isupper()) # Upper is White
            layer = piece_to_num[piece.upper()] # Piece defines layer
            board_state[layer, 7-col, row] = color*2-1 # White=+1, Black=-1
      
      board_states.append(board_state)
      
    return board_states, board_fen

  predict_data = []
  states, fens = pgn_to_states(pgn)
  # print (len(states))

  # Create an array with "move_hist" states
  # that serves as the basis for future state extensions
  state_hist = [np.zeros((6, 8, 8), dtype=np.int8) for i in range(move_hist)]

  # Parse through all the PGN based board states
  # Create arrays of PGN states
  for state in states:
    state_hist.pop(0)
    state_hist.append(state)
    state_save = np.array(state_hist).reshape((6*move_hist, 8, 8))
    if (group=='group_by_piece'):
      state_shuffle = np.zeros((6*move_hist, 8, 8), dtype=np.int8)
      for i in range(6):
        for j in range(move_hist):
          state_shuffle[j+i*move_hist, :, :] = state_save[i+j*6, :, :]
      predict_data.append(state_shuffle)
    else:
      predict_data.append(state_save)
      
  with torch.no_grad():
    predict_data = torch.tensor(predict_data, dtype=torch.int8)
    predict_data = predict_data.to("cpu", dtype=torch.float32)
    with torch.jit.optimized_execution(False):
      model = torch.jit.load('./chess_model.pt', map_location="cpu").eval()
      predict_result = model(predict_data)
    items = dict()    
    for index in range(predict_result.size(dim=0)):
      one_item = dict(\
        move=str(index//2+1)+("W" if index%2==0 else "B"), \
        rating=str(round(predict_result[index].item(),2)), \
        fen=fens[index])
      items.update({str(index):one_item})
    # print(items)
    # mean = torch.mean(y, axis=0)
    # mean = mean.item()
  
  return(items) 
    
'''
# ==========================================
# Test Bench
# ==========================================

# PGNs from Training Set
# ----------------------
# Game Statistics
# Idx   winner  white_elo  black_elo  diff  board_states
# 4959       1       2685       2100   585            49
# 1810      -1       2733       2634    99            88
# 2312       0       2723       2626    97            27

# Winner 1
pgn = "1.d4 Nf6 2.Nf3 g6 3.c4 Bg7 4.Nc3 d6 5.e4 c6 6.Be2 O-O 7.O-O Qe8 8.e5 Nfd7 9.exd6 exd6 10.Bf4 Qe6 11.Re1 Rd8 12.Bd3 Qg4 13.Bxd6 Nf6 14.c5 Bf5 15.h3 Qh5 16.g4 Nxg4 17.Bxf5 gxf5 18.hxg4 fxg4 19.Ne5 f5 20.Qb3+ Kh8 21.Nf7+ Kg8 22.Nxd8+ Kh8 23.Nf7+ Kg8 24.Nh6+ Kh8 25.Qg8#"

# Winner -1
# pgn = "1.e4 e6 2.d4 d5 3.Nd2 Be7 4.e5 c5 5.Qg4 g6 6.dxc5 Nd7 7.Ndf3 Nxc5 8.Bd3 Qa5+ 9.Bd2 Qb6 10.b4 Nxd3+ 11.cxd3 Bd7 12.a4 h5 13.Qd4 Nh6 14.Ne2 Nf5 15.Qb2 O-O 16.O-O Rfc8 17.h3 Qa6 18.b5 Qb6 19.Ned4 Bc5 20.Nxf5 exf5 21.a5 Qxb5 22.Rfb1 Qxb2 23.Rxb2 Bc6 24.Be3 b6 25.d4 Be7 26.axb6 axb6 27.Rxa8 Rxa8 28.Rxb6 Rc8 29.Bg5 Bf8 30.e6 fxe6 31.Ne5 Be8 32.Rxe6 Kh7 33.Ra6 Bg7 34.Bf6 Bxf6 35.Rxf6 Kg7 36.Rd6 Bf7 37.Rd7 Rf8 38.h4 f4 39.Rb7 Kg8 40.Kf1 Be8 41.Ke2 Rf6 42.Kf3 Kf8 43.Rc7 Ba4 44.Rc5 Bd1#"

# Winner 0
# pgn = "1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 5.Ng3 Bg6 6.h4 h6 7.Nf3 Nd7 8.h5 Bh7 9.Bd3 Bxd3 10.Qxd3 Ngf6 11.Bf4 e6 12.O-O-O Be7 13.Ne4 O-O 14.Kb1"

# PGNs from Chess.com
# -------------------

# Winner 1
# pgn = "1. e4 e5 2. f4 exf4 3. Nf3 g5 4. Bc4 g4 5. O-O gxf3 6. Qxf3 Bh6 7. d4 Qh4 8. Nc3 Nc6 9. Nd5 Kd8 10. c3 d6 11. Nxf4 Nge7 12. g3 Qg4 13. Qg2 Bd7 14. h3 Qg8 15. Nh5 Bxc1 16. Raxc1 Qg5 17. g4 Ng6 18. Rce1 Rf8 19. Qg3 Qh4 20. Qxh4+ Nxh4 21. Nf6 Ng6 22. Nxh7 Rh8 23. Ng5 Be8 24. Bxf7 Ke7 25. Kg2 Rf8 26. Bb3 Bd7 27. Kg3 Na5 28. Bf7 Nh8 29. e5 Nxf7 30. exd6+ Kxd6 31. Rf6+ Kd5 32. Nxf7 Nc6 33. b3 Rae8 34. Rc1 Ke4 35. d5 Ne5 36. Re1+ Kxd5 37. Rd1+ Ke4 38. Rf4+ Ke3 39. Re1+ Kd2 40. Rxe5 Bc6 41. Rxe8 Rxe8 42. g5 Re3+ 43. Kh4 Bd7 44. Rd4+"

# Winner -1
# pgn = "1. e4 c5 2. Nf3 e6 3. d4 cxd4 4. Nxd4 Nc6 5. Nb5 d6 6. c4 Nf6 7. N1c3 a6 8. Na3 d5 9. cxd5 exd5 10. exd5 Nb4 11. Be2 Bc5 12. O-O O-O 13. Bf3 Bf5 14. Bg5 Re8 15. Qd2 b5 16. Rad1 Nd3 17. Nab1 h6 18. Bh4 b4 19. Na4 Bd6 20. Bg3 Rc8 21. b3 g5 22. Bxd6 Qxd6 23. g3 Nd7 24. Bg2 Qf6 25. a3 a5 26. axb4 axb4 27. Qa2 Bg6 28. d6 g4 29. Qd2 Kg7 30. f3 Qxd6 31. fxg4 Qd4+ 32. Kh1 Nf6 33. Rf4 Ne4 34. Qxd3 Nf2+ 35. Rxf2 Bxd3 36. Rfd2 Qe3 37. Rxd3 Rc1 38. Nb2 Qf2 39. Nd2 Rxd1+ 40. Nxd1 Re1+"

probability = chess_predict(pgn)
print (probability)
'''