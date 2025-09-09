import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================================================
#
# HELPER FUNCTIONS (train.py/test.pyで必要になる全ての補助関数)
#
# ===================================================================================================

def unique_list(l):
    """重複を除いたリストを返すヘルパー関数"""
    return list(set(l))

def get_all_file_names(directory):
    """ディレクトリ内のファイル名をすべて取得するヘルパー関数"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    """軌道データを絶対座標から各歩行者の初期位置からの相対座標（変位）に変換する"""
    first_values_dict = {}
    for pedestrian in PedsList_seq[0]:
        first_values_dict[pedestrian] = x_seq[0, lookup_seq[pedestrian], :]
    
    for frame_num in range(x_seq.shape[0]):
        for pedestrian in PedsList_seq[frame_num]:
            x_seq[frame_num, lookup_seq[pedestrian], :] -= first_values_dict[pedestrian]
    return x_seq, first_values_dict

def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    """相対座標（変位）から絶対座標に軌道を復元する"""
    for frame_num in range(x_seq.shape[0]):
        for pedestrian in PedsList_seq[frame_num]:
            x_seq[frame_num, lookup_seq[pedestrian], :] += first_values_dict[pedestrian]
    return x_seq

def getCoef(outputs):
    """モデルの出力から二変量正規分布のパラメータ（平均、標準偏差、相関）を抽出する"""
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr

def Gaussian2DLikelihood(outputs, targets, PedsList_seq, lookup_seq):
    """二変量正規分布の負の対数尤度（損失）を計算する"""
    mux, muy, sx, sy, corr = getCoef(outputs)
    x_coords = targets[:, :, 0]
    y_coords = targets[:, :, 1]
    
    # マスクを作成して、フレームに存在しない歩行者の損失を0にする
    mask = torch.zeros_like(mux)
    for frame_num in range(len(PedsList_seq) -1): # 最後のフレームは予測対象ではない
        for ped in PedsList_seq[frame_num + 1]:
             mask[frame_num, lookup_seq[ped]] = 1
    
    # 損失計算
    vx = x_coords - mux
    vy = y_coords - muy
    sx_pow_2 = sx.pow(2)
    sy_pow_2 = sy.pow(2)
    
    z = (vx/sx).pow(2) + (vy/sy).pow(2) - 2*((corr*vx*vy)/(sx*sy))
    
    # detは相関が1または-1の場合に0になりうるので、微小値を追加
    determinant = 1-corr.pow(2)
    epsilon = 1e-20 # ゼロ除算を避けるための微小値
    
    n = torch.log(2*math.pi*sx*sy*torch.sqrt(determinant + epsilon)) + (z/(2*determinant + epsilon))

    loss = torch.sum(n * mask) / torch.sum(mask)
    return loss

def sample_gaussian_2d(mux, muy, sx, sy, corr, PedsList_seq, lookup_seq):
    """二変量正規分布からサンプリングを行う"""
    o_mux, o_muy, o_sx, o_sy, o_corr = mux.data, muy.data, sx.data, sy.data, corr.data
    
    numNodes = mux.size(1)
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)

    for node in range(numNodes):
        mean = torch.Tensor([o_mux[0, node], o_muy[0, node]])
        cov = torch.Tensor([[o_sx[0, node]*o_sx[0, node], o_corr[0, node]*o_sx[0, node]*o_sy[0, node]],
                            [o_corr[0, node]*o_sx[0, node]*o_sy[0, node], o_sy[0, node]*o_sy[0, node]]])
        
        # 多変量正規分布からのサンプリング
        next_pos = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample()
        next_x[node] = next_pos[0]
        next_y[node] = next_pos[1]
    
    return next_x, next_y

def get_mean_error(predicted_trajs, true_trajs, PedsList_seqs, PedsList_seqs_true, use_cuda, lookup_seq):
    """Average Displacement Error (ADE)を計算する"""
    error = 0
    counter = 0
    for i in range(len(PedsList_seqs)): # シーケンスの各フレーム
        for j, pedId in enumerate(PedsList_seqs[i]):
            if pedId in PedsList_seqs_true[i]:
                pred_pos = predicted_trajs[i, lookup_seq[pedId], :]
                true_pos = true_trajs[i, lookup_seq[pedId], :]
                error += torch.dist(pred_pos, true_pos)
                counter += 1
    if counter != 0:
        return error / counter
    else:
        return 0

def get_final_error(predicted_trajs, true_trajs, PedsList_seqs, PedsList_seqs_true, lookup_seq):
    """Final Displacement Error (FDE)を計算する"""
    error = 0
    counter = 0
    # 最後のフレームのみを評価
    last_frame_peds = PedsList_seqs[-1]
    for j, pedId in enumerate(last_frame_peds):
        if pedId in PedsList_seqs_true[-1]:
            pred_pos = predicted_trajs[-1, lookup_seq[pedId], :]
            true_pos = true_trajs[-1, lookup_seq[pedId], :]
            error += torch.dist(pred_pos, true_pos)
            counter += 1
    if counter != 0:
        return error / counter
    else:
        return 0

def time_lr_scheduler(optimizer, epoch, lr_decay=0.95, lr_decay_epoch=10):
    """学習率をスケジューリングする"""
    if epoch % lr_decay_epoch == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay
    return optimizer


def sample_validation_data(x_seq, PedsList_seq, grid_seq, args, net, lookup_seq, numPedsList_seq, dataloader):
    """Validation/Test時に、観測軌道から未来の軌道を予測する"""
    obs_length = args.seq_length - args.pred_length
    
    # 観測シーケンス
    obs_traj = x_seq[:obs_length]
    obs_grid = grid_seq[:obs_length]
    obs_PedsList = PedsList_seq[:obs_length]

    # 最初の損失計算（観測部分のみ）
    numNodes = len(lookup_seq)
    hidden_states = Variable(torch.zeros(numNodes, args.rnn_size)).to(device)
    cell_states = Variable(torch.zeros(numNodes, args.rnn_size)).to(device)

    outputs, _, _ = net(obs_traj, obs_grid, hidden_states, cell_states, obs_PedsList, numPedsList_seq, dataloader, lookup_seq)
    loss = Gaussian2DLikelihood(outputs, x_seq[1:obs_length+1], PedsList_seq[1:obs_length+1], lookup_seq)

    # 予測シーケンスを生成
    ret_x_seq = x_seq.clone()
    
    # 観測の最後の点
    current_pos = x_seq[obs_length-1].clone()

    for i in range(args.pred_length):
        # 現在の位置からグリッドマスクを計算
        current_grid = getSequenceGridMask(current_pos.unsqueeze(0), dataloader.get_dataset_dimension('eth'), [PedsList_seq[obs_length-1+i]], args.neighborhood_size, args.grid_size, args.use_cuda)
        
        # モデルに入力
        outputs, hidden_states, cell_states = net(current_pos.unsqueeze(0), current_grid, hidden_states, cell_states, [PedsList_seq[obs_length-1+i]], numPedsList_seq, dataloader, lookup_seq)
        
        # 出力から次の位置をサンプリング
        mux, muy, sx, sy, corr = getCoef(outputs)
        next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr, PedsList_seq, lookup_seq)
        
        # 次の入力を作成
        next_pos = torch.zeros_like(current_pos)
        for ped_index, ped_id in enumerate(lookup_seq):
            if ped_id in PedsList_seq[obs_length+i]:
                 next_pos[lookup_seq[ped_id], 0] = next_x[lookup_seq[ped_id]]
                 next_pos[lookup_seq[ped_id], 1] = next_y[lookup_seq[ped_id]]
        
        # 予測された軌道に保存
        ret_x_seq[obs_length+i] = next_pos
        current_pos = next_pos.clone()

    return ret_x_seq, loss


# ===================================================================================================
#
# DATALOADER CLASS (修正済みのDataLoaderクラス)
#
# ===================================================================================================

class DataLoader():

    def __init__(self, f_prefix, batch_size=5, seq_length=20, num_of_validation=0, forcePreProcess=False, infer=False, generate=False):
        
        # Social-STGCNNのデータセット構造に直接対応
        base_test_dataset = [
            'eth/test.txt', 'hotel/test.txt', 'zara1/test.txt', 'zara2/test.txt', 'univ/test.txt'
        ]
        base_train_dataset = [
            'eth/train.txt', 'hotel/train.txt', 'zara1/train.txt', 'zara2/train.txt', 'univ/train.txt'
        ]
        base_validation_dataset = [
            'eth/val.txt', 'hotel/val.txt', 'zara1/val.txt', 'zara2/val.txt', 'univ/val.txt'
        ]

        self.f_prefix = f_prefix

        if infer:
            self.data_files = [os.path.join(f_prefix, d) for d in base_test_dataset]
        else:
            self.data_files = [os.path.join(f_prefix, d) for d in base_train_dataset]
        self.data_files = [f for f in self.data_files if os.path.exists(f)] # 存在しないファイルは除外

        self.validation_files = [os.path.join(f_prefix, d) for d in base_validation_dataset]
        self.validation_files = [f for f in self.validation_files if os.path.exists(f)]

        self.infer = infer
        self.generate = generate
        
        if num_of_validation > 0 and len(self.validation_files) > 0:
            self.additional_validation = True
            num_of_validation = np.clip(num_of_validation, 0, len(self.validation_files))
            self.validation_files = random.sample(self.validation_files, num_of_validation)
        else:
            self.additional_validation = False
            if num_of_validation > 0:
                 print("Validation file not found. Continuing without validation.")

        if infer:
            if self.additional_validation:
                self.data_dirs = self.validation_files
            else:
                self.data_dirs = self.data_files
        else:
            self.data_dirs = self.data_files

        self.numDatasets = len(self.data_dirs)
        self.target_ids = []

        self.preprocessed_dir = os.path.join(f_prefix, 'preprocessed')
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        self.data_file_tr = os.path.join(self.preprocessed_dir, "trajectories_train.cpkl")
        self.data_file_te = os.path.join(self.preprocessed_dir, "trajectories_test.cpkl")
        self.data_file_vl = os.path.join(self.preprocessed_dir, "trajectories_val.cpkl")

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.val_fraction = 0

        # Preprocessing
        if self.additional_validation and (not os.path.exists(self.data_file_vl) or forcePreProcess):
            print("Creating pre-processed validation data from raw data")
            self.frame_preprocess(self.validation_files, self.data_file_vl)

        if self.infer and not self.additional_validation and (not os.path.exists(self.data_file_te) or forcePreProcess):
            print("Creating pre-processed test data from raw data")
            self.frame_preprocess(self.data_dirs, self.data_file_te)
        
        if not self.infer and (not os.path.exists(self.data_file_tr) or forcePreProcess):
            print("Creating pre-processed training data from raw data")
            self.frame_preprocess(self.data_dirs, self.data_file_tr)

        # Load preprocessed data
        if self.infer:
            if self.additional_validation:
                self.load_preprocessed(self.data_file_vl, validation_set=True)
            else:
                self.load_preprocessed(self.data_file_te)
        else:
            self.load_preprocessed(self.data_file_tr)

        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)


    def frame_preprocess(self, data_dirs, data_file):
        all_frame_data, frameList_data, numPeds_data, pedsList_data, target_ids, orig_data = [], [], [], [], [], []
        
        for dataset_index, directory in enumerate(data_dirs):
            print("Now processing: ", directory)
            column_names = ['frame_num', 'ped_id', 'x', 'y']
            df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'}, delimiter=r'\s+', header=None, names=column_names)
            
            current_target_ids = np.array(df['ped_id'].unique())
            target_ids.append(current_target_ids)

            data = np.array(df)
            orig_data.append(data)
            data[:, [2, 3]] = data[:, [3, 2]]
            data = data.T

            frameList = sorted(list(np.unique(data[0, :])))
            
            frameList_data.append(frameList)
            numPeds_data.append([])
            all_frame_data.append([])
            pedsList_data.append([])
            
            for frame in frameList:
                pedsInFrame = data[:, data[0, :] == frame]
                pedsList = pedsInFrame[1, :].tolist()
                
                pedsWithPos = []
                for ped in pedsList:
                    current_ped_data = pedsInFrame[:, pedsInFrame[1, :] == ped]
                    current_y, current_x = current_ped_data[2, 0], current_ped_data[3, 0]
                    pedsWithPos.append([ped, current_x, current_y])

                all_frame_data[dataset_index].append(np.array(pedsWithPos))
                pedsList_data[dataset_index].append(pedsList)
                numPeds_data[dataset_index].append(len(pedsList))
            
        valid_frame_data = [[] for _ in range(len(data_dirs))]
        valid_numPeds_data = [[] for _ in range(len(data_dirs))]
        valid_pedsList_data = [[] for _ in range(len(data_dirs))]

        with open(data_file, "wb") as f:
            pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_numPeds_data, valid_frame_data, pedsList_data, valid_pedsList_data, target_ids, orig_data), f, protocol=2)


    def load_preprocessed(self, data_file, validation_set=False):
        print(f"Loading {'validation' if validation_set else 'train or test'} dataset: {data_file}")
        with open(data_file, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.data, self.frameList, self.numPedsList = self.raw_data[0], self.raw_data[1], self.raw_data[2]
        self.valid_data = self.raw_data[4] if self.raw_data[4] else [[] for _ in self.data]
        self.valid_numPedsList = self.raw_data[3] if self.raw_data[3] else [[] for _ in self.data]
        self.pedsList, self.valid_pedsList = self.raw_data[5], self.raw_data[6] if self.raw_data[6] else [[] for _ in self.data]
        self.target_ids, self.orig_data = self.raw_data[7], self.raw_data[8]

        counter, valid_counter = 0, 0
        print('Sequence size(frame) ------>', self.seq_length)

        for dataset in range(len(self.data)):
            dataset_name = self.data_dirs[dataset].replace(self.f_prefix, '')
            num_seq = len(self.data[dataset]) // self.seq_length
            counter += num_seq
            print(f"{'Validation' if validation_set else 'Training'} data from {dataset_name}: {len(self.data[dataset])} frames, {num_seq} sequences")

        self.num_batches = int(counter / self.batch_size)
        self.valid_num_batches = int(valid_counter / self.batch_size)
        print(f"Total number of {'validation' if validation_set else 'training'} batches: {self.num_batches}")


    def next_batch(self):
        x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids_batch = [], [], [], [], [], []
        i = 0
        while i < self.batch_size:
            if not self.data or self.dataset_pointer >= len(self.data): self.tick_batch_pointer()
            
            frame_data = self.data[self.dataset_pointer]
            idx = self.frame_pointer
            
            if idx + self.seq_length + 1 <= len(frame_data):
                x_batch.append(frame_data[idx:idx + self.seq_length])
                y_batch.append(frame_data[idx + 1:idx + self.seq_length + 1])
                numPedsList_batch.append(self.numPedsList[self.dataset_pointer][idx:idx + self.seq_length])
                PedsList_batch.append(self.pedsList[self.dataset_pointer][idx:idx + self.seq_length])
                
                seq_index = idx // self.seq_length
                current_target_ids = self.target_ids[self.dataset_pointer]
                # target_idはシーケンス内の最初の歩行者IDを仮に使う
                target_id = PedsList_batch[-1][0][0] if PedsList_batch[-1] and PedsList_batch[-1][0] else -1
                target_ids_batch.append(target_id)
                
                d.append(self.dataset_pointer)
                self.frame_pointer += self.seq_length
                i += 1
            else:
                self.tick_batch_pointer()
        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids_batch

    def tick_batch_pointer(self, valid=False):
        if not valid:
            self.dataset_pointer += 1
            if self.dataset_pointer >= len(self.data): self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer += 1
            if self.valid_dataset_pointer >= len(self.valid_data): self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
        
    def reset_batch_pointer(self, valid=False):
        if not valid:
            self.dataset_pointer, self.frame_pointer = 0, 0
        else:
            self.valid_dataset_pointer, self.valid_frame_pointer = 0, 0
    
    def convert_proper_array(self, x_seq, num_pedlist, pedlist):
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
        lookup_table = {val: i for i, val in enumerate(unique_ids)}
        seq_data = np.zeros((self.seq_length, len(lookup_table), 2))
        
        for i, frame in enumerate(x_seq):
            if frame.size > 0:
                p_ids = frame[:, 0].astype(int)
                indices = [lookup_table[p_id] for p_id in p_ids]
                seq_data[i, indices, :] = frame[:, 1:3]
                
        return Variable(torch.from_numpy(seq_data).float()), lookup_table

    def get_directory_name_with_pointer(self, pointer_index):
        path = self.data_dirs[pointer_index]
        return path.split('/')[-2]

    def get_dataset_dimension(self, key):
        dataset_dimensions = {
            'eth': [720, 576], 'hotel': [720, 576], 'zara1': [720, 576],
            'zara2': [720, 576], 'univ': [720, 576]
        }
        return dataset_dimensions.get(key, [720, 576])

