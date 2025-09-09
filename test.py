import torch
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os
import time
import pickle

from model import SocialModel
from utils import DataLoader, get_mean_error, get_final_error, revert_seq, vectorize_seq
from grid import getSequenceGridMask

def main():
    parser = argparse.ArgumentParser()
    
    # --- 必ず指定が必要な引数 ---
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file (.tar)')

    # --- データセットとシーケンスに関する引数 ---
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='Root directory of the datasets')
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observation length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Prediction length')
    
    # --- その他 ---
    parser.add_argument('--forcePreProcess', action="store_true", default=False,
                        help='Force preprocess the data again')

    args = parser.parse_args()
    test(args)


def test(args):
    # 学習時の設定ファイルを読み込む
    # モデルのディレクトリパスを取得
    model_dir = os.path.dirname(args.model_path)
    config_path = os.path.join(model_dir, 'config.pkl')
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'rb') as f:
        saved_args = pickle.load(f)

    # モデルのインスタンス化
    net = SocialModel(saved_args, infer=True)
    if saved_args.use_cuda:
        net = net.to(device)
    
    # 学習済み重みの読み込み
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval() # 評価モードに設定
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # データローダーの準備
    # テスト時は観測+予測の全長をseq_lengthとする
    seq_length = args.obs_length + args.pred_length
    dataloader = DataLoader(args.data_root, batch_size=1, seq_length=seq_length, forcePreProcess=args.forcePreProcess, infer=True)

    # エラー計算用のリスト
    total_ade, total_fde = 0, 0
    total_sequences = 0

    print("Starting testing...")
    # 全てのテストデータセットをループ
    for i in range(dataloader.numDatasets):
        dataloader.reset_batch_pointer()
        dataloader.dataset_pointer = i # データセットを順番に指定

        dataset_ade, dataset_fde = 0, 0
        num_sequences_in_dataset = dataloader.num_batches

        # 現在のデータセットの全シーケンスをループ
        for batch in range(num_sequences_in_dataset):
            start_time = time.time()
            
            # バッチサイズ1でデータを取得
            x, _, d, numPedsList, PedsList, _ = dataloader.next_batch()
            
            # バッチ内のシーケンスを取得
            x_seq, d_seq, PedsList_seq = x[0], d[0], PedsList[0]

            # データをテンソルに変換
            x_seq_tensor, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList[0], PedsList_seq)
            
            # 評価のために元の絶対座標を保持
            orig_x_seq = x_seq_tensor.clone()
            
            # モデルへの入力のために相対座標に変換
            vec_x_seq, first_values_dict = vectorize_seq(x_seq_tensor.clone(), PedsList_seq, lookup_seq)
            
            if saved_args.use_cuda:
                vec_x_seq = vec_x_seq.to(device)

            # ----- 軌道予測 -----
            numNodes = len(lookup_seq)
            hidden_states = Variable(torch.zeros(numNodes, saved_args.rnn_size)).to(device)
            cell_states = Variable(torch.zeros(numNodes, saved_args.rnn_size)).to(device)
            
            # 予測結果を格納するテンソル
            predicted_traj = vec_x_seq.clone()

            # 1. 観測フェーズ: 8フレームをモデルに入力し、隠れ状態を更新
            for frame_num in range(args.obs_length - 1):
                current_grid = getSequenceGridMask(vec_x_seq[frame_num].unsqueeze(0), dataloader.get_dataset_dimension('eth'), [PedsList_seq[frame_num]], saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)
                _, hidden_states, cell_states = net(vec_x_seq[frame_num].unsqueeze(0), current_grid, hidden_states, cell_states, [PedsList_seq[frame_num]], numPedsList[0], dataloader, lookup_seq)

            # 2. 予測フェーズ: 12フレームを自己回帰的に予測
            # 観測の最後のフレームが、予測の最初の入力になる
            current_pos = vec_x_seq[args.obs_length - 1].unsqueeze(0)
            
            for frame_num in range(args.pred_length):
                current_grid = getSequenceGridMask(current_pos, dataloader.get_dataset_dimension('eth'), [PedsList_seq[args.obs_length - 1 + frame_num]], saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)
                
                outputs, hidden_states, cell_states = net(current_pos, current_grid, hidden_states, cell_states, [PedsList_seq[args.obs_length - 1 + frame_num]], numPedsList[0], dataloader, lookup_seq)
                
                # 出力から次の位置をサンプリング
                mux, muy, sx, sy, corr = getCoef(outputs)
                next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr, [PedsList_seq[args.obs_length - 1 + frame_num]], lookup_seq)

                # 次の入力を作成
                next_pos = torch.zeros_like(current_pos.squeeze(0))
                for ped_id in PedsList_seq[args.obs_length + frame_num]:
                     if ped_id in lookup_seq:
                        next_pos[lookup_seq[ped_id], 0] = next_x[lookup_seq[ped_id]]
                        next_pos[lookup_seq[ped_id], 1] = next_y[lookup_seq[ped_id]]
                
                # 予測結果を保存し、次の入力とする
                predicted_traj[args.obs_length + frame_num] = next_pos
                current_pos = next_pos.unsqueeze(0)

            # ----- エラー計算 -----
            # 予測軌道を絶対座標に戻す
            predicted_traj_abs = revert_seq(predicted_traj, PedsList_seq, lookup_seq, first_values_dict)
            
            # 予測部分のみを切り出す
            pred_true_traj = orig_x_seq[args.obs_length:]
            pred_predicted_traj = predicted_traj_abs[args.obs_length:]
            
            # ADEとFDEを計算
            ade = get_mean_error(pred_predicted_traj.data, pred_true_traj.data, PedsList_seq[args.obs_length:], PedsList_seq[args.obs_length:], saved_args.use_cuda, lookup_seq)
            fde = get_final_error(pred_predicted_traj.data, pred_true_traj.data, PedsList_seq[args.obs_length:], PedsList_seq[args.obs_length:], lookup_seq)

            dataset_ade += ade
            dataset_fde += fde
            
            end_time = time.time()
            print(f"Dataset: {dataloader.data_dirs[i].split('/')[-2]}, Sequence: {batch + 1}/{num_sequences_in_dataset}, ADE: {ade:.4f}, FDE: {fde:.4f}, Time: {end_time - start_time:.2f}s")
        
        # データセットごとの平均エラー
        if num_sequences_in_dataset > 0:
            avg_dataset_ade = dataset_ade / num_sequences_in_dataset
            avg_dataset_fde = dataset_fde / num_sequences_in_dataset
            print(f"--- Average for {dataloader.data_dirs[i].split('/')[-2]} --- ADE: {avg_dataset_ade:.4f}, FDE: {avg_dataset_fde:.4f} ---")
            total_ade += dataset_ade
            total_fde += dataset_fde
            total_sequences += num_sequences_in_dataset

    # 全データセットでの平均エラー
    if total_sequences > 0:
        final_ade = total_ade / total_sequences
        final_fde = total_fde / total_sequences
        print("\n======================= FINAL RESULTS =======================")
        print(f"Average ADE across all test datasets: {final_ade:.4f}")
        print(f"Average FDE across all test datasets: {final_fde:.4f}")
        print("===========================================================")

if __name__ == '__main__':
    main()
