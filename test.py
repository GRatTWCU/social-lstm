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
    
    # Parse the parameters
    sample_args = parser.parse_args()
    
    # for drive run
    prefix = ''
    f_prefix = '.'
    if sample_args.drive is True:
        prefix = 'drive/semester_project/social_lstm_final/'
        f_prefix = 'drive/semester_project/social_lstm_final'

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
                print(f"Warning: target_id is not a scalar integer: {target_id}, type: {type(target_id)}")
                # 強制的にスカラーに変換
                if hasattr(target_id, '__iter__') and not isinstance(target_id, str):
                    try:
                        target_id = next(iter(target_id))
                    except StopIteration:
                        print(f"Error: target_id iterator is empty. Skipping batch {batch}.")
                        continue
                if hasattr(target_id, 'item'):
                    target_id = target_id.item()
                try:
                    target_id = int(target_id)
                except ValueError:
                    print(f"Error: Could not convert target_id {target_id} to int. Skipping batch {batch}.")
                    continue
                print(f"Forced conversion to: {target_id}")

            try:
                # Debugging: Check x_seq before cleaning
                # print(f"DEBUG: Before cleaning - x_seq shape: {x_seq.shape if hasattr(x_seq, 'shape') else 'N/A'}")
                # print(f"DEBUG: Before cleaning - x_seq has NaN: {np.isnan(x_seq).any() if hasattr(x_seq, 'any') else 'N/A'}")

                dataloader.clean_test_data(x_seq, target_id, sample_args.obs_length, sample_args.pred_length)
                dataloader.clean_ped_list(x_seq, PedsList_seq, target_id, sample_args.obs_length, sample_args.pred_length)

                # Debugging: Check x_seq after cleaning
                # print(f"DEBUG: After cleaning - x_seq shape: {x_seq.shape if hasattr(x_seq, 'shape') else 'N/A'}")
                # print(f"DEBUG: After cleaning - x_seq has NaN: {np.isnan(x_seq).any() if hasattr(x_seq, 'any') else 'N/A'}")

            except Exception as e:
                print(f"Error in data cleaning for batch {batch}: {e}")
                print(f"x_seq type: {type(x_seq)}")
                print(f"target_id: {target_id} (type: {type(target_id)})")
                if hasattr(x_seq, 'shape'):
                    print(f"x_seq shape: {x_seq.shape}")
                print(f"Skipping batch {batch}")
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
                    print(f"Warning: target_id {target_id} not found in lookup_seq. Available keys: {list(lookup_seq.keys())}")
                    target_id = list(lookup_seq.keys())[0]
                    print(f"Using first available target_id: {target_id}")
                else:
                    print(f"Warning: Empty lookup_seq for batch {batch}, skipping batch.")
                    continue
            
            # will be used for error calculation
            orig_x_seq = x_seq.clone() 
            
            # Check if target_id is valid in orig_x_seq after lookup_seq mapping
            if lookup_seq[target_id] >= orig_x_seq.shape[1]:
                print(f"Error: Mapped index {lookup_seq[target_id]} for target_id {target_id} is out of bounds for orig_x_seq (shape: {orig_x_seq.shape}). Skipping batch {batch}.")
                continue

            target_id_values = orig_x_seq[0][lookup_seq[target_id], 0:2]
            
            # grid mask calculation
            if sample_args.method == 2:  # obstacle lstm
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda, True)
            elif sample_args.method == 1:  # social lstm   
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda) # Assuming this is the correct call
            else: # For vanilla LSTM, grid_seq might not be needed or handled differently
                grid_seq = None # Or handle appropriately for method 3

            # vectorize datapoints
            x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)
            if sample_args.use_cuda:
                x_seq = x_seq.to(device)

            # The sample function
            if sample_args.method == 3:  # vanilla lstm
                # Extract the observed part of the trajectories
                obs_traj, obs_PedsList_seq = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length]
                ret_x_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru)
            else:
                # Extract the observed part of the trajectories
                if grid_seq is None: # Handle case where grid_seq might be None for certain methods
                    print(f"Warning: grid_seq is None for method {sample_args.method}. This might be expected for some methods, but check if it's causing issues.")
                    obs_traj, obs_PedsList_seq = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length]
                    ret_x_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru)
                else:
                    obs_traj, obs_PedsList_seq, obs_grid = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length], grid_seq[:sample_args.obs_length]
                    ret_x_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru, grid=grid_seq) # Pass grid_seq to sample function

            # Debugging: Check ret_x_seq after sampling
            if ret_x_seq is None or ret_x_seq.nelement() == 0:
                print(f"Warning: ret_x_seq is empty or None after sampling for batch {batch}. Skipping error calculation for this batch.")
                continue
            if torch.isnan(ret_x_seq).any():
                print(f"Warning: ret_x_seq contains NaN values after sampling for batch {batch}. Skipping error calculation for this batch.")
                continue

            # During test, we just want to calculate ADE and FDE
            # ADE (Average Displacement Error)
            # FDE (Final Displacement Error)
            
            # Get the predicted trajectory for the target_id
            # Ensure lookup_seq[target_id] is a valid index for ret_x_seq
            if lookup_seq[target_id] >= ret_x_seq.shape[1]:
                print(f"Error: Mapped index {lookup_seq[target_id]} for target_id {target_id} is out of bounds for ret_x_seq (shape: {ret_x_seq.shape}). Skipping batch {batch}.")
                continue

            # Assuming ret_x_seq has shape [pred_length, num_peds, 2]
            pred_traj_target = ret_x_seq[:, lookup_seq[target_id], 0:2]

            # Get the ground truth trajectory for the target_id
            # Assuming orig_x_seq has shape [seq_length, num_peds, 2]
            gt_traj_target = orig_x_seq[sample_args.obs_length:, lookup_seq[target_id], 0:2]

            # Ensure both predicted and ground truth trajectories are not empty
            if pred_traj_target.nelement() == 0 or gt_traj_target.nelement() == 0:
                print(f"Warning: Predicted or ground truth trajectory for target_id {target_id} in batch {batch} is empty. Skipping error calculation.")
                continue

            mean_error = get_mean_error(pred_traj_target, gt_traj_target)
            final_error_val = get_final_error(pred_traj_target, gt_traj_target)

            if not torch.isnan(mean_error) and not torch.isnan(final_error_val):
                total_error += mean_error
                final_error += final_error_val
                num_batches_processed += 1
            else:
                print(f"Warning: NaN detected in error calculation for batch {batch}. Mean error: {mean_error}, Final error: {final_error_val}. Skipping this batch for overall error accumulation.")

            end = time.time()
            # print('Time taken for batch', batch, 'is', end - start)

            # Store results for submission and plotting
            # submission_preprocess and result_preprocess need to be defined or imported
            # Assuming submission_preprocess and result_preprocess are available from helper.py or similar
            # If not, these lines will cause errors and need to be implemented or removed based on your project structure.
            submission_data = submission_preprocess(dataloader, ret_x_seq.data.cpu().numpy(), sample_args.pred_length, sample_args.obs_length, target_id)
            iteration_submission.append(submission_data)
            result_data = result_preprocess(dataloader, ret_x_seq.data.cpu().numpy(), orig_x_seq.data.cpu().numpy(), sample_args.pred_length, sample_args.obs_length, target_id)
            iteration_result.append(result_data)

        # Calculate average errors for the iteration
        if num_batches_processed > 0:
            avg_ade = total_error / num_batches_processed
            avg_fde = final_error / num_batches_processed
            print(f"Iteration: {iteration + 1}  ADE (prediction part) mean error: {avg_ade}")
            print(f"Iteration: {iteration + 1}  FDE (prediction part) final error: {avg_fde}")
        else:
            print(f"Iteration: {iteration + 1}  No valid batches processed for error calculation. ADE and FDE are NaN.")
            avg_ade = torch.tensor(float('nan'))
            avg_fde = torch.tensor(float('nan'))

        if not torch.isnan(avg_ade) and avg_ade < smallest_err:
            smallest_err = avg_ade
            smallest_err_iter_num = iteration + 1
            submission_store = iteration_submission
            result_store = iteration_result
        
        # Reset batch pointer for the next iteration
        dataloader.reset_batch_pointer()

    print("Smallest error iteration:", smallest_err_iter_num)

    # Save submission and plot files
    # This part assumes submission_store and result_store are populated correctly.
    # If submission_preprocess and result_preprocess were commented out, these will be empty.
    # Ensure these functions are implemented and correctly populate the stores if you need this output.
    save_submission_file(submission_store, result_directory, model_name, dataloader.get_all_directory_namelist())
    save_plot_file(result_store, plot_directory, plot_test_file_directory, dataloader.get_all_directory_namelist())

    print("テスト完了:", time.time() - start_time, "秒") # Assuming start_time is defined at the beginning of main()

    print("=== 総実行時間:", time.time() - total_start_time, "秒 (", (time.time() - total_start_time) / 60, "分) ===") # Assuming total_start_time is defined at the very beginning of the script.


if __name__ == '__main__':
    # Define total_start_time at the very beginning of script execution
    total_start_time = time.time()
    main()
