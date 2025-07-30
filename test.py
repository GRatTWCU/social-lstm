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
from helper import *
from grid import getSequenceGridMask, getGridMask

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
    dataloader = DataLoader(f_prefix, 1, seq_length, forcePreProcess=True, infer=True)
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

        for batch in range(dataloader.num_batches):
            start = time.time()
            # Get data
            x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()

            # Get the sequence
            x_seq, d_seq, numPedsList_seq, PedsList_seq, target_id = x[0], d[0], numPedsList[0], PedsList[0], target_ids[0]
            
            # target_idの型変換を確実に行う
            print(f"Original target_id type: {type(target_id)}, value: {target_id}")
            
            # target_idが配列やリストの場合の処理
            if isinstance(target_id, (list, tuple)):
                if len(target_id) > 0:
                    target_id = target_id[0]
                else:
                    print("Warning: Empty target_id list, skipping batch")
                    continue
            elif isinstance(target_id, np.ndarray):
                if target_id.size > 0:
                    target_id = target_id.flatten()[0]
                else:
                    print("Warning: Empty target_id array, skipping batch")
                    continue
            elif hasattr(target_id, 'item'):
                target_id = target_id.item()
            
            # 最終的にスカラー値に変換
            try:
                if isinstance(target_id, (np.floating, np.integer)):
                    target_id = float(target_id)
                target_id = int(float(target_id))
                print(f"Converted target_id: {target_id}")
            except (ValueError, TypeError) as e:
                print(f"Error converting target_id: {e}, skipping batch")
                continue
            
            # データのクリーニング前にデバッグ情報を出力
            print(f"x_seq shape: {x_seq.shape if hasattr(x_seq, 'shape') else 'No shape'}")
            print(f"target_id: {target_id} (type: {type(target_id)})")
            
            try:
                dataloader.clean_test_data(x_seq, target_id, sample_args.obs_length, sample_args.pred_length)
                dataloader.clean_ped_list(x_seq, PedsList_seq, target_id, sample_args.obs_length, sample_args.pred_length)
            except Exception as e:
                print(f"Error in data cleaning: {e}")
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
                    print("Warning: Empty lookup_seq, skipping batch")
                    continue
            
            # will be used for error calculation
            orig_x_seq = x_seq.clone() 
            
            target_id_values = orig_x_seq[0][lookup_seq[target_id], 0:2]
            
            # grid mask calculation
            if sample_args.method == 2:  # obstacle lstm
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda, True)
            elif sample_args.method == 1:  # social lstm   
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)

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
                obs_traj, obs_PedsList_seq, obs_grid = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length], grid_seq[:sample_args.obs_length]
                ret_x_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq, saved_args, dataset_data, dataloader, lookup_seq, numPedsList_seq, sample_args.gru, obs_grid)
            
            # revert the points back to original space
            ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)
            
            # 【修正】予測部分の正しい評価
            # ADE: 予測部分（obs_length以降）の平均変位誤差
            # FDE: 予測部分の最終時刻での変位誤差
            pred_start = sample_args.obs_length
            pred_end = sample_args.obs_length + sample_args.pred_length
            
            # シーケンス長の確認
            if pred_end <= ret_x_seq.shape[0] and pred_end <= orig_x_seq.shape[0]:
                try:
                    # 予測部分のADE計算
                    ade = get_mean_error(
                        ret_x_seq[pred_start:pred_end].data, 
                        orig_x_seq[pred_start:pred_end].data, 
                        PedsList_seq[pred_start:pred_end], 
                        PedsList_seq[pred_start:pred_end], 
                        sample_args.use_cuda, 
                        lookup_seq
                    )
                    
                    # 予測部分の最終時刻でのFDE計算
                    fde = get_final_error(
                        ret_x_seq[pred_end-1:pred_end].data, 
                        orig_x_seq[pred_end-1:pred_end].data, 
                        PedsList_seq[pred_end-1:pred_end], 
                        PedsList_seq[pred_end-1:pred_end], 
                        lookup_seq
                    )
                    
                    # NaN値でない場合のみ加算
                    if not torch.isnan(ade) and not torch.isnan(fde):
                        total_error += ade
                        final_error += fde
                
                except Exception as e:
                    print(f"Error calculating ADE/FDE: {e}")
                    continue

            end = time.time()

            if (batch + 1) % 500 == 0:
                print('Current file : ', dataloader.get_file_name(0), ' Processed trajectory number : ', batch+1, 'out of', dataloader.num_batches, 'trajectories in time', end - start)

            if dataset_pointer_ins is not dataloader.dataset_pointer:
                if dataloader.dataset_pointer != 0:
                    iteration_submission.append(submission)
                    iteration_result.append(results)

                dataset_pointer_ins = dataloader.dataset_pointer
                submission = []
                results = []

            submission.append(submission_preprocess(dataloader, ret_x_seq.data[sample_args.obs_length:, lookup_seq[target_id], :].cpu().numpy(), sample_args.pred_length, sample_args.obs_length, target_id))
            results.append((x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq, lookup_seq, dataloader.get_frame_sequence(seq_length), target_id, sample_args.obs_length))

        iteration_submission.append(submission)
        iteration_result.append(results)

        submission_store.append(iteration_submission)
        result_store.append(iteration_result)

        # 最終結果の計算
        avg_ade = total_error / dataloader.num_batches if dataloader.num_batches > 0 else float('inf')
        avg_fde = final_error / dataloader.num_batches if dataloader.num_batches > 0 else float('inf')

        if total_error < smallest_err:
            print("**********************************************************")
            print('Best iteration has been changed. Previous best iteration: ', smallest_err_iter_num+1, 'Error: ', smallest_err / dataloader.num_batches if dataloader.num_batches > 0 else float('inf'))
            print('New best iteration : ', iteration+1, 'Error: ', avg_ade)
            smallest_err_iter_num = iteration
            smallest_err = total_error

        # 【修正】正しいラベルで出力
        print('Iteration:', iteration+1, ' ADE (prediction part) mean error: ', avg_ade)
        print('Iteration:', iteration+1, ' FDE (prediction part) final error: ', avg_fde)

    print('Smallest error iteration:', smallest_err_iter_num+1)
    
    if smallest_err_iter_num >= 0:
        dataloader.write_to_file(submission_store[smallest_err_iter_num], result_directory, prefix, model_name)
        dataloader.write_to_plot_file(result_store[smallest_err_iter_num], os.path.join(plot_directory, plot_test_file_directory))


def sample(x_seq, Pedlist, args, net, true_x_seq, true_Pedlist, saved_args, dimensions, dataloader, look_up, num_pedlist, is_gru, grid=None):
    '''
    The sample function
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    with torch.no_grad():
        # Construct variables for hidden and cell states
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.to(device)
        if not is_gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.to(device)
        else:
            cell_states = None

        ret_x_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, 2))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.to(device)

        # For the observed part of the trajectory
        for tstep in range(args.obs_length-1):
            if grid is None:  # vanilla lstm
                # Do a forward prop
                out_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            else:
                # Do a forward prop
                out_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), [grid[tstep]], hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_obs)
            # Sample from the bivariate Gaussian
            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)
            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y

        ret_x_seq[:args.obs_length, :, :] = x_seq.clone()

        # Last seen grid
        if grid is not None:  # no vanilla lstm
            prev_grid = grid[-1].clone()

        # For the predicted part of the trajectory
        for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
            # Do a forward prop
            if grid is None:  # vanilla lstm
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, 2), hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            else:
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, 2), [prev_grid], hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs)
            # Sample from the bivariate Gaussian
            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)

            # Store the predicted position
            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y

            # List of x_seq at the last time-step (assuming they exist until the end)
            true_Pedlist[tstep+1] = [int(_x_seq) for _x_seq in true_Pedlist[tstep+1]]
            next_ped_list = true_Pedlist[tstep+1].copy()
            converted_pedlist = [look_up[_x_seq] for _x_seq in next_ped_list]
            list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))

            if args.use_cuda:
                list_of_x_seq = list_of_x_seq.to(device)
           
            # Get their predicted positions
            current_x_seq = torch.index_select(ret_x_seq[tstep+1], 0, list_of_x_seq)

            if grid is not None:  # no vanilla lstm
                # Compute the new grid masks with the predicted positions
                if args.method == 2:  # obstacle lstm
                    prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(true_Pedlist[tstep+1]), saved_args.neighborhood_size, saved_args.grid_size, True)
                elif args.method == 1:  # social lstm   
                    prev_grid = getGridMask(current_x_seq.data.cpu(), dimensions, len(true_Pedlist[tstep+1]), saved_args.neighborhood_size, saved_args.grid_size)

                prev_grid = Variable(torch.from_numpy(prev_grid).float())
                if args.use_cuda:
                    prev_grid = prev_grid.to(device)

        return ret_x_seq


def submission_preprocess(dataloader, ret_x_seq, pred_length, obs_length, target_id):
    seq_length = pred_length + obs_length

    # begin and end index of obs. frames in this seq.
    begin_obs = (dataloader.frame_pointer - seq_length)
    end_obs = (dataloader.frame_pointer - pred_length)

    # get original data for frame number and ped ids
    observed_data = dataloader.orig_data[dataloader.dataset_pointer][begin_obs:end_obs, :]
    frame_number_predicted = dataloader.get_frame_sequence(pred_length)
    ret_x_seq_c = ret_x_seq.copy()
    ret_x_seq_c[:, [0, 1]] = ret_x_seq_c[:, [1, 0]]  # x, y -> y, x
    repeated_id = np.repeat(target_id, pred_length)  # add id
    id_integrated_prediction = np.append(repeated_id[:, None], ret_x_seq_c, axis=1)
    frame_integrated_prediction = np.append(frame_number_predicted[:, None], id_integrated_prediction, axis=1)  # add frame number
    result = np.append(observed_data, frame_integrated_prediction, axis=0)

    return result


if __name__ == '__main__':
    main()
