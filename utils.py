import os
import pickle
import numpy as np

class DataLoader():

    def __init__(self, args, logging):
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.pred_length = args.pred_length
        self.class_balance = args.class_balance
        self.force_preprocessing = args.force_preprocessing
        self.logging = logging

        self.train_data_file = os.path.join(self.data_dir, self.dataset, "train_data.npz")
        self.val_data_file = os.path.join(self.data_dir, self.dataset, "val_data.npz")
        self.test_data_file = os.path.join(self.data_dir, self.dataset, "test_data.npz")

        self.data_dir = os.path.join(self.data_dir, self.dataset)

        if not (os.path.exists(self.train_data_file)) or self.force_preprocessing:
            self.logging.info("Preprocessing data...")
            self.preprocess(os.path.join(self.data_dir, 'train.txt'), self.train_data_file)
            self.preprocess(os.path.join(self.data_dir, 'val.txt'), self.val_data_file)
            self.preprocess(os.path.join(self.data_dir, 'test.txt'), self.test_data_file)

        self.logging.info("Loading preprocessed data...")
        train_data = np.load(self.train_data_file, allow_pickle=True)
        val_data = np.load(self.val_data_file, allow_pickle=True)
        test_data = np.load(self.test_data_file, allow_pickle=True)

        self.train_x, self.train_y, self.train_class = train_data['x'], train_data['y'], train_data['class']
        self.val_x, self.val_y, self.val_class = val_data['x'], val_data['y'], val_data['class']
        self.test_x, self.test_y = test_data['x'], test_data['y']

        self.logging.info(
            "Train data loaded. Shape of x: {}, y: {}, class: {}".format(self.train_x.shape, self.train_y.shape,
                                                                          self.train_class.shape))

        # create pointers
        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0

        self.num_batches = int(self.train_x.shape[0] / self.batch_size)
        self.val_num_batches = int(self.val_x.shape[0] / self.batch_size)
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

    def next_batch(self, split='train'):

        if split == 'train':
            x_batch = self.train_x[self.train_pointer:self.train_pointer + self.batch_size, :, :]
            y_batch = self.train_y[self.train_pointer:self.train_pointer + self.batch_size, :]
            class_batch = self.train_class[self.train_pointer:self.train_pointer + self.batch_size]
            self.train_pointer += self.batch_size
        elif split == 'val':
            x_batch = self.val_x[self.val_pointer:self.val_pointer + self.batch_size, :, :]
            y_batch = self.val_y[self.val_pointer:self.val_pointer + self.batch_size, :]
            class_batch = self.val_class[self.val_pointer:self.val_pointer + self.batch_size]
            self.val_pointer += self.batch_size

        return x_batch, y_batch, class_batch

    def next_test_batch(self, end_of_epoch):
        '''
        A strange way to extract test batch for visualization
        :return:
        '''
        if not end_of_epoch:
            x_batch = self.test_x[self.test_pointer, :, :, :]
            y_batch = self.test_y[self.test_pointer, :, :]
            self.test_pointer += 1
        else:
            return None, None
        return np.reshape(x_batch, (1, x_batch.shape[0], x_batch.shape[1], x_batch.shape[2])), \
               np.reshape(y_batch, (1, y_batch.shape[0], y_batch.shape[1]))

    def reset_batch_pointer(self, split='train', logging=True):
        if split == 'train':
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
                rand_idx = np.random.permutation(self.train_x.shape[0])
                self.train_x = self.train_x[rand_idx, :, :]
                self.train_y = self.train_y[rand_idx, :]
                self.train_class = self.train_class[rand_idx]
            if logging: self.logging.info("Reshuffle training data")
            self.train_pointer = 0

        elif split == 'val':
            self.val_pointer = 0
        elif split == 'test':
            self.test_pointer = 0

    def preprocess(self, data_file, out_file):
        '''
        Function that processes the data and saves it in a numpy file
        :param data_file:
        :param out_file:
        :return:
        '''
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
        last_frame = frame_data[frame_id]
        neighbors = []
        for ped_data in last_frame:
            # Get the sequence of this pedestrian
            ped_id = ped_data[1]
            try:
                ped_seq = self.get_ped_sequence(frame_data, ped_id, frame_id, seq_length)
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
                frame = frame_data[i]
                ped_data = frame[frame[:, 1] == ped_id][0]
                ped_seq.append(ped_data[2:4])
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
                frame = frame_data[frame_id + i]
                ped_data = frame[frame[:, 1] == ped_id][0]
                ped_seq.append(ped_data[2:4])
            except:
                return None
        return np.array(ped_seq)
    
    # ---------------- ここが修正された関数です ----------------
    # ---------------- 修正された関数です ----------------
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
            if len(x_seq[frame_num]) > 0:
                # x, y座標のNaN値チェック（列インデックス1, 2）
                nan_mask_x = np.isnan(x_seq[frame_num][:, 1])
                nan_mask_y = np.isnan(x_seq[frame_num][:, 2])
                nan_mask = nan_mask_x | nan_mask_y
                
                # NaN要素を除去
                x_seq[frame_num] = x_seq[frame_num][~nan_mask]
        
        # predicted part でtarget_id以外を除去
        for frame_num in range(obs_length, obs_length + predicted_length):
            if len(x_seq[frame_num]) > 0:
                # デバッグ情報
                if frame_num == obs_length:  # 最初の予測フレームのみ
                    print(f"DEBUG: Frame {frame_num} shape: {x_seq[frame_num].shape}")
                    print(f"DEBUG: Frame {frame_num} ped_ids: {x_seq[frame_num][:, 0]}")
                
                try:
                    # target_idと一致する要素のみ保持
                    target_mask = x_seq[frame_num][:, 0] == target_id
                    x_seq[frame_num] = x_seq[frame_num][target_mask]
                    
                except (ValueError, IndexError, TypeError) as e:
                    print(f"Error processing predicted frame {frame_num}: {e}")
                    print(f"x_seq[{frame_num}] shape: {x_seq[frame_num].shape if len(x_seq[frame_num]) > 0 else 'empty'}")
                    print(f"target_id: {target_id} (type: {type(target_id)})")
                    # エラーが発生した場合、そのフレームを空にする
                    x_seq[frame_num] = np.array([]).reshape(0, 3)
