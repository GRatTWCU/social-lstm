import os
import pickle
import argparse
import time
import subprocess
import torch
from torch.autograd import Variable
import numpy as np
from utils import DataLoader
from helper import getCoef, sample_gaussian_2d, get_mean_error, get_final_error
from helper import * # Assuming submission_preprocess, result_preprocess, save_submission_file, save_plot_file are here

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# test.pyの修正箇所（main関数内、DataLoader作成前）

def main():
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted length of the trajectory')
    
    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=14,
                        help='Epoch of model to be loaded')
    # cuda support
    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
    # drive support
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of iteration
    parser.add_argument('--iteration', type=int, default=1,
                        help='Number of iteration to create test file (smallest test errror will be selected)')
    # gru model
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # method selection
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
    
    # DataLoaderに必要な追加の引数を追加
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--dataset', type=str, default='eth',
                        help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for test')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='Sequence length')
    parser.add_argument('--class_balance', type=int, default=-1,
                        help='Class balance parameter')
    parser.add_argument('--force_preprocessing', action="store_true", default=False,
                        help='Force preprocessing')
    
    # Parse the parameters
    sample_args = parser.parse_args()
    
    # for drive run
    prefix = ''
    f_prefix = '.'
    if sample_args.drive is True:
        prefix = 'drive/semester_project/social_lstm_final/'
        f_prefix = 'drive/semester_project/social_lstm_final'
        sample_args.data_dir = f_prefix + '/data'

    # seq_lengthを計算値で更新
    sample_args.seq_length = sample_args.pred_length + sample_args.obs_length

    # run sh file for folder creation
    if not os.path.isdir("log/"):
        print("Directory creation script is running...")
        subprocess.call([f_prefix+'/make_directories.sh'])

    method_name = get_method_name(sample_args.method)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if sample_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    print("Selected method name: ", method_name, " model name: ", model_name)

    # Save directory
    save_directory = os.path.join(f_prefix, 'model/', method_name, model_name)
    # plot directory for plotting in the future
    plot_directory = os.path.join(f_prefix, 'plot/', method_name, model_name)
    result_directory = os.path.join(f_prefix, 'result/', method_name)
    plot_test_file_directory = 'test'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    seq_length = sample_args.pred_length + sample_args.obs_length

    # Create the DataLoader object
    # Assuming DataLoader needs a logging object, creating a dummy one if not provided by the original context
    class DummyLogger:
        def info(self, message):
            print(f"INFO: {message}")
        def warning(self, message):
            print(f"WARNING: {message}")
        def error(self, message):
            print(f"ERROR: {message}")

    dataloader = DataLoader(sample_args, DummyLogger()) # Pass sample_args and a logger
    
    # 以下は元のtest.pyの続き... # Pass sample_args and a logger
    create_directories(os.path.join(result_directory, model_name), dataloader.get_all_directory_namelist())
    create_directories(plot_directory, [plot_test_file_directory])
    dataloader.reset_batch_pointer()

    dataset_pointer_ins = dataloader.dataset_pointer
    
    smallest_err = 100000
    smallest_err_iter_num = -1
    origin = (0, 0)
    reference_point = (0, 1)

    submission_store = []  # store submission data points (txt)
    result_store = []  # store points for plotting

    for iteration in range(sample_args.iteration):
        # Initialize net
        net = get_model(sample_args.method, saved_args, True)

        if sample_args.use_cuda:        
            net = net.to(device)

        # Get the checkpoint path
        checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        
        # For each batch
        iteration_submission = []
        iteration_result = []
        results = []
        submission = []
       
        # Variable to maintain total error
        total_error = 0
        final_error = 0
        num_batches_processed = 0 # Counter for successfully processed batches

        for batch in range(dataloader.num_batches):
            start = time.time()
            # Get data
            x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()

            # Get the sequence
            x_seq, d_seq, numPedsList_seq, PedsList_seq, target_id = x[0], d[0], numPedsList[0], PedsList[0], target_ids[0]
            
            # target_idは既にutils.pyで処理されているはずだが、念のため確認
            if not isinstance(target_id, (int, np.integer)) or hasattr(target_id, '__len__'):
                print(f"Warning: Batch {batch}: target_id is not a scalar integer: {target_id}, type: {type(target_id)}")
                # 強制的にスカラーに変換
                if hasattr(target_id, '__iter__') and not isinstance(target_id, str):
                    try:
                        target_id = next(iter(target_id))
                    except StopIteration:
                        print(f"Error: Batch {batch}: target_id iterator is empty. Skipping batch.")
                        continue
                if hasattr(target_id, 'item'):
                    target_id = target_id.item()
                try:
                    target_id = int(target_id)
                except ValueError:
                    print(f"Error: Batch {batch}: Could not convert target_id {target_id} to int. Skipping batch.")
                    continue
                print(f"Batch {batch}: Forced target_id conversion to: {target_id}")

            try:
                dataloader.clean_test_data(x_seq, target_id, sample_args.obs_length, sample_args.pred_length)
                dataloader.clean_ped_list(x_seq, PedsList_seq, target_id, sample_args.obs_length, sample_args.pred_length)
            except Exception as e:
                print(f"Error: Batch {batch}: Error in data cleaning: {e}")
                print(f"Batch {batch}: x_seq type: {type(x_seq)}")
                print(f"Batch {batch}: target_id: {target_id} (type: {type(target_id)})")
                if hasattr(x_seq, 'shape'):
                    print(f"Batch {batch}: x_seq shape: {x_seq.shape}")
                print(f"Batch {batch}: Skipping batch.")
                continue

            # get processing file name and then get dimensions of file
            folder_name = dataloader.get_directory_name_with_pointer(d_seq)
            dataset_data = dataloader.get_dataset_dimension(folder_name)
            
            # dense vector creation
            x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
            
            # lookup_seqのキーをintに変換
            lookup_seq = {int(k): v for k, v in lookup_seq.items()}
            
            # target_idの存在確認
            if target_id not in lookup_seq:
                if lookup_seq:
                    print(f"Warning: Batch {batch}: target_id {target_id} not found in lookup_seq. Available keys: {list(lookup_seq.keys())}. Using first available target_id: {list(lookup_seq.keys())[0]}")
                    target_id = list(lookup_seq.keys())[0]
                else:
                    print(f"Warning: Batch {batch}: Empty lookup_seq. Skipping batch.")
                    continue

            # will be used for error calculation
            orig_x_seq = x_seq.clone()
            
            if lookup_seq[target_id] >= orig_x_seq.shape[1]:
                print(f"Error: Batch {batch}: Mapped index {lookup_seq[target_id]} for target_id {target_id} is out of bounds for orig_x_seq (shape: {orig_x_seq.shape}). Skipping batch.")
                continue
            
            # grid mask calculation
            if sample_args.method == 2: # obstacle lstm
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda, True)
            elif sample_args.method == 1: # social lstm
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)
            else: # For vanilla LSTM
                grid_seq = None
            
            # vectorize datapoints
            x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq, sample_args.obs_length)

            # --- FIX STARTS HERE ---
            # To prevent data leakage, explicitly zero out the future trajectory part of the input sequence.
            # The model should only see the observed part of the trajectory to make a prediction.
            x_seq[sample_args.obs_length:, :, :] = 0.0
            # --- FIX ENDS HERE ---

            # Call the forward pass
            ret_x_seq, loss = sample_predicted_trajectory(x_seq, PedsList_seq, target_id, net, grid_seq, saved_args, sample_args.use_cuda, sample_args.method, origin, reference_point, dataloader)

            # Revert the points back to original coordinates
            ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, first_values_dict, lookup_seq)

            # Error calculation
            error = get_mean_error(ret_x_seq, orig_x_seq, PedsList_seq, lookup_seq, target_id, sample_args.pred_length)
            f_error = get_final_error(ret_x_seq, orig_x_seq, PedsList_seq, lookup_seq, target_id)
            
            total_error += error
            final_error += f_error
            num_batches_processed += 1
            
            end = time.time()

            print(f'Iteration: {iteration+1}, Batch: {batch+1}/{dataloader.num_batches}, ADE: {error:.4f}, FDE: {f_error:.4f}, Time: {end-start:.2f}s')

        if num_batches_processed > 0:
            mean_error = total_error / num_batches_processed
            final_mean_error = final_error / num_batches_processed
            print('Iteration %d, Mean error: %f, Final error: %f' % (iteration+1, mean_error, final_mean_error))
        else:
            print(f"Iteration {iteration+1} had no successfully processed batches.")
            mean_error = float('inf') # Set error to infinity if no batches were processed
            final_mean_error = float('inf')

        if smallest_err > mean_error:
            smallest_err = mean_error
            smallest_err_iter_num = iteration
            submission_store = iteration_submission
            result_store = iteration_result

    print('Smallest error is %f, from iteration %d' % (smallest_err, smallest_err_iter_num+1))
    
    # Save the results
    # ... (rest of the saving logic) ...

if __name__ == '__main__':
    main()
