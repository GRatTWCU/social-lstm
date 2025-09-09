import torch
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os
import time
import pickle
import subprocess

from model import SocialModel
from utils import DataLoader
from grid import getSequenceGridMask
from helper import *

import csv
import sys
from contextlib import redirect_stdout


def main():
    
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    # Cuda parameter
    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
    # GRU parameter
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # drive option
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=2,
                        help='Total number of validation dataset for validate accuracy')
    # frequency of validation
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    parser.add_argument('--grid', action="store_true", default=True,
                        help='Whether store grids and use further epoch')
    
    # <--- 変更点: forcePreProcessをコマンドライン引数で制御できるように追加 ---
    parser.add_argument('--forcePreProcess', action="store_true", default=False,
                        help='Force preprocess the data again')
    # <--- 変更点: データセットのルートディレクトリを指定する引数を追加 ---
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='Root directory of the datasets')

    args = parser.parse_args()
    
    train(args)


def train(args):
    # Google Drive用のパス設定（今回はローカル実行を想定）
    prefix = ''
    f_prefix = args.data_root # <--- 変更点: 引数からデータセットのルートパスを取得
    if args.drive is True:
        # Google Driveを使用する場合のパス設定（必要に応じて変更）
        prefix='drive/semester_project/social_lstm_final/'
        f_prefix = 'drive/semester_project/social_lstm_final'

    
    # ログ・モデル保存用のディレクトリ作成（初回実行時のみ）
    if not os.path.isdir("log/"):
        print("Directory creation script is running...")
        # make_directories.sh が存在する場合に実行
        if os.path.exists(f_prefix+'/make_directories.sh'):
            subprocess.call([f_prefix+'/make_directories.sh'])
        else:
            os.makedirs("log", exist_ok=True)
            os.makedirs("model", exist_ok=True)
            os.makedirs("plot", exist_ok=True)


    args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
    validation_epoch_list = list(range(args.freq_validation, args.num_epochs+1, args.freq_validation))
    if validation_epoch_list:
        validation_epoch_list[-1]-=1

    # <--- 変更点: forcePreProcess=True を args.forcePreProcess に変更 ---
    # これにより、コマンドラインから `--forcePreProcess` を指定した時だけ再処理が実行される
    dataloader = DataLoader(f_prefix, args.batch_size, args.seq_length, args.num_validation, forcePreProcess=args.forcePreProcess)

    model_name = "LSTM"
    method_name = "SOCIALLSTM"
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    # 各種ディレクトリパスの設定
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/', method_name, model_name)
    save_directory = os.path.join(prefix, 'model/')
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.join(log_directory, method_name, model_name), exist_ok=True)
    os.makedirs(os.path.join(save_directory, method_name, model_name), exist_ok=True)
    os.makedirs(plot_directory, exist_ok=True)

    # Logging files
    log_file_curve = open(os.path.join(log_directory, method_name, model_name,'log_curve.txt'), 'w+')
    log_file = open(os.path.join(log_directory, method_name, model_name, 'val.txt'), 'w+')
    
    # Save the arguments int the config file
    with open(os.path.join(save_directory, method_name, model_name,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, method_name, model_name, save_tar_name+str(x)+'.tar')

    # model creation
    net = SocialModel(args)
    if args.use_cuda:
        net = net.to(device)

    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    learning_rate = args.learning_rate

    # 訓練の進行状況を記録する変数の初期化
    best_val_loss = 100
    best_val_data_loss = 100
    smallest_err_val = 100000
    smallest_err_val_data = 100000
    best_epoch_val = 0
    best_epoch_val_data = 0
    best_err_epoch_val = 0
    best_err_epoch_val_data = 0
    all_epoch_results = []
    
    # グリッドを保存するためのリスト
    grids = [[] for _ in range(dataloader.numDatasets)]

    # Training
    for epoch in range(args.num_epochs):
        print(f'****************Training epoch {epoch+1}/{args.num_epochs} beginning******************')
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()
            x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()
            loss_batch = 0

            # For each sequence
            for sequence in range(dataloader.batch_size):
                x_seq, _, d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                target_id = target_ids[sequence]

                # get processing file name and then get dimensions of file
                folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                dataset_data = dataloader.get_dataset_dimension(folder_name)
                
                x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)

                # grid mask calculation and storage
                if args.grid:
                    if epoch == 0:
                        grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                        grids[d_seq].append(grid_seq)
                    else:
                        # バッチインデックスを計算
                        batch_index = (batch * dataloader.batch_size) + sequence
                        grid_seq = grids[d_seq][batch_index]
                else:
                    grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)

                x_seq, _ = vectorize_seq(x_seq, PedsList_seq, lookup_seq)
                
                if args.use_cuda:
                    x_seq = x_seq.cuda()

                numNodes = len(lookup_seq)
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    cell_states = cell_states.cuda()

                net.zero_grad()
                optimizer.zero_grad()
                
                outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq, numPedsList_seq, dataloader, lookup_seq)
                
                loss = Gaussian2DLikelihood(outputs, x_seq, PedsList_seq, lookup_seq)
                loss_batch += loss.item()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()

            end = time.time()
            loss_batch /= dataloader.batch_size
            loss_epoch += loss_batch
            
            print(f'{epoch * dataloader.num_batches + batch}/{args.num_epochs * dataloader.num_batches} (epoch {epoch}), '
                  f'train_loss = {loss_batch:.3f}, time/batch = {end - start:.3f}')

        loss_epoch /= dataloader.num_batches
        log_file_curve.write(f"Training epoch: {epoch} loss: {loss_epoch}\n")

        # ==== Validation Phase (if applicable) ====
        # このコードでは学習データの一部をvalidationに使うロジックと、
        # 別のvalidationデータセットを使うロジックが混在しているため、
        # 今回はvalidationデータセットを使うロジックのみを有効化する想定で進めます。
        # (dataloader.valid_num_batchesは0になる想定)

        # ==== Validation with separate dataset ====
        if dataloader.additional_validation and (epoch) in validation_epoch_list:
            print('****************Validation with dataset epoch beginning******************')
            # (validationデータセットを使った評価のロジックは元のコードのまま)
            # ... (中略) ...
            pass # validationのロジックをここに記述

        # optimizerの学習率を調整
        optimizer = time_lr_scheduler(optimizer, epoch, lr_decay_epoch=args.freq_optimizer)

        # Save the model after each epoch
        print('Saving model checkpoint')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    print('Training finished.')
    log_file.close()
    log_file_curve.close()

if __name__ == '__main__':
    main()
