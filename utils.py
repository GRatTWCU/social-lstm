import os
import pickle
import numpy as np
import torch

class DataLoader():

    def __init__(self, args, logging, test_mode=False):
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.pred_length = args.pred_length
        self.class_balance = args.class_balance
        self.force_preprocessing = args.force_preprocessing
        self.logging = logging
        self.test_mode = test_mode

        # train.pyで使用される属性
        self.additional_validation = False
        self.dataset_pointer = 0
        self.valid_num_batches = 0

        self.train_data_file = os.path.join(self.data_dir, self.dataset, "train_data.npz")
        self.val_data_file = os.path.join(self.data_dir, self.dataset, "val_data.npz")
        self.test_data_file = os.path.join(self.data_dir, self.dataset, "test_data.npz")

        self.data_dir = os.path.join(self.data_dir, self.dataset)

        if not self.test_mode:
            if not (os.path.exists(self.train_data_file)) or self.force_preprocessing:
                self.logging.info("Preprocessing data...")
                if os.path.exists(os.path.join(self.data_dir, 'train.txt')):
                    self.preprocess(os.path.join(self.data_dir, 'train.txt'), self.train_data_file)
                if os.path.exists(os.path.join(self.data_dir, 'val.txt')):
                    self.preprocess(os.path.join(self.data_dir, 'val.txt'), self.val_data_file)
                if os.path.exists(os.path.join(self.data_dir, 'test.txt')):
                    self.preprocess(os.path.join(self.data_dir, 'test.txt'), self.test_data_file)

            if os.path.exists(self.train_data_file):
                self.logging.info("Loading preprocessed data...")
                train_data = np.load(self.train_data_file, allow_pickle=True)
                val_data = np.load(self.val_data_file, allow_pickle=True) if os.path.exists(self.val_data_file) else None
                test_data = np.load(self.test_data_file, allow_pickle=True) if os.path.exists(self.test_data_file) else None

                self.train_x, self.train_y, self.train_class = train_data['x'], train_data['y'], train_data.get('class_id', train_data.get('class', np.zeros(len(train_data['x']))))
                
                if val_data is not None:
                    self.val_x, self.val_y, self.val_class = val_data['x'], val_data['y'], val_data.get('class_id', val_data.get('class', np.zeros(len(val_data['x']))))
                else:
                    # バリデーションデータがない場合、訓練データの一部を使用
                    split_idx = int(0.8 * len(self.train_x))
                    self.val_x, self.val_y, self.val_class = self.train_x[split_idx:], self.train_y[split_idx:], self.train_class[split_idx:]
                    self.train_x, self.train_y, self.train_class = self.train_x[:split_idx], self.train_y[:split_idx], self.train_class[:split_idx]
                
                if test_data is not None:
                    self.test_x, self.test_y = test_data['x'], test_data['y']
                else:
                    self.test_x, self.test_y = self.val_x, self.val_y

                self.logging.info(
                    "Train data loaded. Shape of x: {}, y: {}, class: {}".format(self.train_x.shape, self.train_y.shape,
                                                                                  self.train_class.shape))
        else:
            # テストモード用の最小限の初期化
            self.train_x = self.train_y = self.train_class = np.array([])
            self.val_x = self.val_y = self.val_class = np.array([])
            self.test_x = self.test_y = []

        # create pointers
        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0

        if not self.test_mode and hasattr(self, 'train_x') and len(self.train_x) > 0:
            self.num_batches = int(self.train_x.shape[0] / self.batch_size)
            self.valid_num_batches = int(self.val_x.shape[0] / self.batch_size) if hasattr(self, 'val_x') and len(self.val_x) > 0 else 0
        else:
            self.num_batches = 0
            self.valid_num_batches = 0

        self.reset_batch_pointer(split='train', logging=False)
        self.reset_batch_pointer(split='val', logging=False)
        self.reset_batch_pointer(split='test', logging=False)

        # For social pooling
        if self.dataset == 'eth' or self.dataset == 'hotel':
            self.grid_size = 4
        elif self.dataset == 'zara1' or self.dataset == 'zara2':
            self.grid_size = 4
        elif self.dataset == 'univ':
            self.grid_size = 4
        else:
            self.grid_size = 4

    def get_len_of_dataset(self):
        """データセットの数を返す"""
        return 1

    def switch_to_dataset_type(self, load_data=True):
        """データセットタイプを切り替える（train.pyで使用）"""
        pass

    def get_directory_name_with_pointer(self, d_seq):
        """ポインタでディレクトリ名を取得"""
        return self.dataset

    def get_dataset_dimension(self, folder_name):
        """データセットの次元情報を返す"""
        return {
            'width': 1920,
            'height': 1080,
            'left': 0,
            'top': 0
        }

    def convert_proper_array(self, x_seq, numPedsList_seq, PedsList_seq):
        """配列を適切な形式に変換"""
        # 簡単な実装 - 実際のプロジェクトに応じて調整が必要
        lookup_seq = {}
        if len(PedsList_seq) > 0 and len(PedsList_seq[0]) > 0:
            for i, ped_id in enumerate(PedsList_seq[0]):
                lookup_seq[int(ped_id)] = i
        
        # x_seqをTensorに変換
        if isinstance(x_seq, np.ndarray):
            x_seq = torch.FloatTensor(x_seq)
        elif not isinstance(x_seq, torch.Tensor):
            x_seq = torch.FloatTensor(np.array(x_seq))
        
        return x_seq, lookup_seq

    def get_file_name(self, offset):
        """ファイル名を取得"""
        return f"{self.dataset}.txt"

    def get_frame_sequence(self, seq_length):
        """フレームシーケンスを取得"""
        return list(range(seq_length))

    def write_to_plot_file(self, results, path):
        """プロット用ファイルに書き込み"""
        # プレースホルダー実装
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(os.path.join(path, 'results.txt'), 'w') as f:
            f.write(f"Results written with {len(results)} datasets\n")

    def get_all_directory_namelist(self):
        """すべてのディレクトリ名のリストを返す"""
        return [self.dataset]

    def clean_ped_list(self, x_seq, PedsList_seq, target_id, obs_length, pred_length):
        """PedsList_seqをクリーンアップ"""
        # observed partは変更なし
        # predicted partはtarget_idのみ保持
        for frame_num in range(obs_length, obs_length + pred_length):
            if frame_num < len(PedsList_seq):
                # target_idのみを保持
                if target_id in PedsList_seq[frame_num]:
                    PedsList_seq[frame_num] = [target_id]
                else:
                    PedsList_seq[frame_num] = []

    def next_batch(self, split='train'):
        """次のバッチを取得"""
        if split == 'train':
            if hasattr(self, 'train_x') and len(self.train_x) > 0:
                x_batch = self.train_x[self.train_pointer:self.train_pointer + self.batch_size]
                y_batch = self.train_y[self.train_pointer:self.train_pointer + self.batch_size]
                class_batch = self.train_class[self.train_pointer:self.train_pointer + self.batch_size]
                
                # train.pyが期待する形式に変換
                # x, y, d, numPedsList, PedsList, target_ids の形式で返す
                d = [0] * len(x_batch)  # dataset indices
                numPedsList = []
                PedsList = []
                target_ids = []
                
                for i, x in enumerate(x_batch):
                    # 各シーケンスに対してダミーデータを生成
                    num_peds = x.shape[0] if len(x.shape) > 1 else 1
                    numPedsList.append([num_peds] * self.seq_length)
                    PedsList.append([[j for j in range(num_peds)] for _ in range(self.seq_length)])
                    target_ids.append(0)  # デフォルトのtarget_id
                
                self.train_pointer += self.batch_size
                return x_batch, y_batch, d, numPedsList, PedsList, target_ids
            else:
                return [], [], [], [], [], []
                
        elif split == 'val':
            if hasattr(self, 'val_x') and len(self.val_x) > 0:
                x_batch = self.val_x[self.val_pointer:self.val_pointer + self.batch_size]
                y_batch = self.val_y[self.val_pointer:self.val_pointer + self.batch_size]
                class_batch = self.val_class[self.val_pointer:self.val_pointer + self.batch_size]
                
                # train.pyが期待する形式に変換
                d = [0] * len(x_batch)
                numPedsList = []
                PedsList = []
                target_ids = []
                
                for i, x in enumerate(x_batch):
                    num_peds = x.shape[0] if len(x.shape) > 1 else 1
                    numPedsList.append([num_peds] * self.seq_length)
                    PedsList.append([[j for j in range(num_peds)] for _ in range(self.seq_length)])
                    target_ids.append(0)
                
                self.val_pointer += self.batch_size
                return x_batch, y_batch, d, numPedsList, PedsList, target_ids
            else:
                return [], [], [], [], [], []

    def next_valid_batch(self):
        """バリデーション用の次のバッチを取得"""
        return self.next_batch(split='val')

    def next_test_batch(self, end_of_epoch):
        '''
        A strange way to extract test batch for visualization
        :return:
        '''
        if not end_of_epoch:
            if hasattr(self, 'test_x') and isinstance(self.test_x, list) and len(self.test_x) > 0:
                if self.test_pointer < len(self.test_x):
                    x_batch = self.test_x[self.test_pointer]
                    y_batch = self.test_y[self.test_pointer] if hasattr(self, 'test_y') and len(self.test_y) > self.test_pointer else []
                    self.test_pointer += 1
                    return x_batch, y_batch
            return None, None
        else:
            return None, None

    def reset_batch_pointer(self, split='train', logging=True, valid=False):
        """バッチポインタをリセット"""
        if valid:
            split = 'val'
            
        if split == 'train':
            if hasattr(self, 'train_x') and len(self.train_x) > 0:
                if self.class_balance != -1:
                    # class balance
                    self.logging.info("Perform class balance with ratio {}".format(self.class_balance))
                    class_ids = np.where(self.train_class == 1)[0]
                    other_ids = np.where(self.train_class == 0)[0]
                    # oversample the minor class
                    if len(class_ids) * self.class_balance < len(other_ids):
                        class_ids = np.random.choice(class_ids, size=int(len(other_ids)/self.class_balance), replace=True)
                    random_idx = np.random.permutation(len(other_ids) + len(class_ids))
                    all_ids = np.concatenate((class_ids, other_ids))
                    all_ids = all_ids[random_idx]
                    self.train_x = self.train_x[all_ids]
                    self.train_y = self.train_y[all_ids]
                    self.train_class = self.train_class[all_ids]
                    self.num_batches = int(len(all_ids)/self.batch_size)
                else:
                    if len(self.train_x) > 0:
                        rand_idx = np.random.permutation(self.train_x.shape[0])
                        self.train_x = self.train_x[rand_idx]
                        self.train_y = self.train_y[rand_idx]
                        self.train_class = self.train_class[rand_idx]
                if logging: 
                    self.logging.info("Reshuffle training data")
            self.train_pointer = 0

        elif split == 'val':
            self.val_pointer = 0
        elif split == 'test':
            self.test_pointer = 0

    # 既存のメソッドはそのまま保持
    def preprocess(self, data_file, out_file):
        '''
        Function that processes the data and saves it in a numpy file
        :param data_file:
        :param out_file:
        :return:
        '''
        if not os.path.exists(data_file):
            self.logging.warning(f"Data file {data_file} not found. Skipping preprocessing.")
            return
            
        all_x_but_one = []
        all_y_but_one = []

        all_class_but_one = []
        data = np.loadtxt(data_file, delimiter=' ')
        # remove nan frames
        data = data[~np.isnan(data).any(axis=1)]

        frame_ids = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame_id in frame_ids:
            frame_data.append(data[data[:, 0] == frame_id, :])

        num_peds = np.unique(data[:, 1]).tolist()
        all_sequences = []
        all_peds_sequences = {} # ped_id -> [seq_len, 2]
        for ped in num_peds:
            # extract trajectories of each pedestrian
            ped_data = data[data[:, 1] == ped, :]
            all_peds_sequences[ped] = ped_data[:, [2,3]]

        # get all fixed-length sequences of all pedestrians
        for ped in num_peds:
            sequences = self.get_all_sequences(all_peds_sequences[ped], self.seq_length)
            all_sequences.extend(sequences)

        all_sequences = np.array(all_sequences)
        x = all_sequences[:, 0:self.seq_length - self.pred_length, :]
        y = all_sequences[:, -1, :]

        # Find social contexts
        all_x = []
        all_y = []
        all_class = []

        self.logging.info("Total number of sequences is {}".format(x.shape[0]))
        for i in range(x.shape[0]):
            if i % 100 == 0:
                self.logging.info("Processed {}/{} sequences".format(i, x.shape[0]))
            seq_x = x[i]
            seq_y = y[i]
            # find the frame id of the last frame of the sequence
            last_frame_data = seq_x[-1] # the last frame of the observed sequence
            # find the frame id of this frame in the original data
            frame_id = self.find_frame_id(frame_data, last_frame_data)
            neighbors = self.get_neighbors(frame_data, frame_id, self.seq_length - self.pred_length)

            # if no neighbors, this is a linear trajectory
            if len(neighbors) == 0:
                all_class.append(0)
            else:
                all_class.append(1)

            # The first dimension is the agent, the rest are neighbors
            all_x.append(np.reshape(np.concatenate((np.reshape(seq_x, (1, seq_x.shape[0], seq_x.shape[1])), neighbors)), (-1, seq_x.shape[0], seq_x.shape[1])))
            all_y.append(seq_y)

        # Save the arrays
        self.logging.info("Shape of all_x: {}".format(np.shape(all_x)))
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        np.savez(out_file, x=np.array(all_x), y=np.array(all_y), class_id=np.array(all_class))
        self.logging.info("Saved preprocessed data to {}".format(out_file))

    def get_all_sequences(self, ped_data, seq_length):
        '''
        Given a pedestrian's trajectory, get all fixed-length sequences
        :param ped_data:
        :param seq_length:
        :return:
        '''
        sequences = []
        for i in range(ped_data.shape[0] - seq_length + 1):
            sequences.append(ped_data[i:i + seq_length, :])
        return sequences

    def find_frame_id(self, frame_data, last_frame_data):
        for i, frame in enumerate(frame_data):
            for ped_data in frame:
                if np.array_equal(ped_data[2:4], last_frame_data):
                    return i
        return -1 # Not found

    def get_neighbors(self, frame_data, frame_id, seq_length):
        '''
        Get neighbors of the agent in the last frame of the observed sequence
        Neighbors are defined as pedestrians who are in the same frame as the agent in the last frame of the observed sequence
        :param frame_data:
        :param frame_id:
        :param seq_length:
        :return:
        '''
        if frame_id == -1 or frame_id >= len(frame_data):
            return np.array([])
            
        last_frame = frame_data[frame_id]
        neighbors = []
        for ped_data in last_frame:
            # Get the sequence of this pedestrian
            ped_id = ped_data[1]
            try:
                ped_seq = self.get_ped_sequence(frame_data, ped_id, frame_id, seq_length)
                if ped_seq is not None:
                    neighbors.append(ped_seq)
            except: # this pedestrian does not have a trajectory of the given length
                continue
        return np.array(neighbors)

    def get_ped_sequence(self, frame_data, ped_id, frame_id, seq_length):
        '''
        Get the sequence of a pedestrian given the last frame id
        :param frame_data:
        :param ped_id:
        :param frame_id:
        :param seq_length:
        :return:
        '''
        ped_seq = []
        for i in range(frame_id - seq_length + 1, frame_id + 1):
            try:
                if i < 0 or i >= len(frame_data):
                    return None
                frame = frame_data[i]
                ped_data = frame[frame[:, 1] == ped_id]
                if len(ped_data) == 0:
                    return None
                ped_seq.append(ped_data[0][2:4])
            except:
                # This pedestrian is not in this frame
                # Return None, so this pedestrian will be discarded
                return None
        return np.array(ped_seq)

    def get_test_data(self, data_file, obs_length, pred_length, delim='\t'):
        '''
        Get test data from the given file
        This function is used for visualizing the test data, so it returns the data in a different format
        :param data_file:
        :param obs_length:
        :param pred_length:
        :param delim:
        :return:
        '''
        if not os.path.exists(data_file):
            self.logging.warning(f"Test data file {data_file} not found.")
            self.test_x = []
            self.test_y = []
            self.test_target_id = []
            self.num_test_batches = 0
            return
            
        seq_length = obs_length + pred_length
        data = np.loadtxt(data_file, delimiter=delim)
        # remove nan frames
        data = data[~np.isnan(data).any(axis=1)]

        frame_ids = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame_id in frame_ids:
            frame_data.append(data[data[:, 0] == frame_id, :])

        all_x = []
        all_y = []
        all_target_id = []

        for frame_id in range(len(frame_data) - seq_length):
            # for each frame, get all pedestrians in this frame
            peds_in_frame = np.unique(frame_data[frame_id][:, 1])
            for ped in peds_in_frame:
                # get the sequence of this pedestrian
                ped_seq = self.get_ped_sequence_test(frame_data, ped, frame_id, seq_length)
                if ped_seq is None:
                    continue
                # Get neighbors of this pedestrian in the observed sequence
                x_seq = []
                for i in range(obs_length):
                    frame = frame_data[frame_id+i]
                    neighbors = frame[frame[:, 1] != ped, :]
                    x_seq.append(np.concatenate((np.reshape(ped_seq[i, :], (1, 2)), neighbors[:, 2:4]), axis=0))
                all_x.append(x_seq)
                all_y.append(ped_seq[obs_length:, :])
                all_target_id.append(ped)

        self.test_x = all_x
        self.test_y = all_y
        self.test_target_id = all_target_id
        self.num_test_batches = len(all_x)

    def get_ped_sequence_test(self, frame_data, ped_id, frame_id, seq_length):
        ped_seq = []
        for i in range(seq_length):
            try:
                if frame_id + i >= len(frame_data):
                    return None
                frame = frame_data[frame_id + i]
                ped_data = frame[frame[:, 1] == ped_id]
                if len(ped_data) == 0:
                    return None
                ped_seq.append(ped_data[0][2:4])
            except:
                return None
        return np.array(ped_seq)
    
    def clean_test_data(self, x_seq, target_id, obs_length, predicted_length):
        """remove (pedid, x , y) array if x or y is nan for each frame in observed part"""
        
        # target_idを確実にスカラーに変換
        if hasattr(target_id, '__iter__') and not isinstance(target_id, str):
            if isinstance(target_id, (list, tuple)):
                target_id = target_id[0] if len(target_id) > 0 else 0
            elif isinstance(target_id, np.ndarray):
                target_id = target_id.flat[0] if target_id.size > 0 else 0
            else:
                target_id = next(iter(target_id))
        
        if hasattr(target_id, 'item'):
            target_id = target_id.item()
        
        # 確実にPythonのintに変換
        target_id = int(float(target_id))
        
        print(f"DEBUG clean_test_data: target_id = {target_id} (type: {type(target_id)})")
        
        # observed part のNaN要素を除去
        for frame_num in range(obs_length):
            if frame_num < len(x_seq) and len(x_seq[frame_num]) > 0:
                # x, y座標のNaN値チェック（列インデックス1, 2）
                if x_seq[frame_num].shape[1] >= 3:  # ped_id, x, y の形式を想定
                    nan_mask_x = np.isnan(x_seq[frame_num][:, 1])
                    nan_mask_y = np.isnan(x_seq[frame_num][:, 2])
                    nan_mask = nan_mask_x | nan_mask_y
                    
                    # NaN要素を除去
                    x_seq[frame_num] = x_seq[frame_num][~nan_mask]
        
        # predicted part でtarget_id以外を除去
        for frame_num in range(obs_length, min(obs_length + predicted_length, len(x_seq))):
            if len(x_seq[frame_num]) > 0:
                # デバッグ情報
                if frame_num == obs_length:  # 最初の予測フレームのみ
                    print(f"DEBUG: Frame {frame_num} shape: {x_seq[frame_num].shape}")
                    if x_seq[frame_num].shape[1] > 0:
                        print(f"DEBUG: Frame {frame_num} ped_ids: {x_seq[frame_num][:, 0]}")
                
                try:
                    # target_idと一致する要素のみ保持
                    if x_seq[frame_num].shape[1] >= 3:  # ped_id, x, y の形式を想定
                        target_mask = x_seq[frame_num][:, 0] == target_id
                        x_seq[frame_num] = x_seq[frame_num][target_mask]
                    
                except (ValueError, IndexError, TypeError) as e:
                    print(f"Error processing predicted frame {frame_num}: {e}")
                    print(f"x_seq[{frame_num}] shape: {x_seq[frame_num].shape if len(x_seq[frame_num]) > 0 else 'empty'}")
                    print(f"target_id: {target_id} (type: {type(target_id)})")
                    # エラーが発生した場合、そのフレームを空にする
                    if len(x_seq[frame_num].shape) == 2 and x_seq[frame_num].shape[1] >= 3:
                        x_seq[frame_num] = np.array([]).reshape(0, 3)
                    else:
                        x_seq[frame_num] = np.array([])
