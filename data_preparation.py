import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

folder_f8_dyn = 'dane/f8/dyn'
folder_f8_stat = 'dane/f8/stat'
folder_f10_dyn = 'dane/f10/dyn'
folder_f10_stat = 'dane/f10/stat'

output_folder_npy = 'dane/prepared_data_npy'
output_folder_csv = 'dane/prepared_data_csv'
os.makedirs(output_folder_npy, exist_ok=True)
os.makedirs(output_folder_csv, exist_ok=True)

def load_files_from_folder(folder, prefix):
    files = os.listdir(folder)
    files = [f for f in files if f.startswith(prefix) and f.endswith('.csv')]
    full_paths = [os.path.join(folder, f) for f in files]
    return full_paths

def loading_all_files():
    train_files_f8 = load_files_from_folder(folder_f8_stat, 'f8_stat_')
    train_files_f10 = load_files_from_folder(folder_f10_stat, 'f10_stat_')
    train_files = train_files_f8 + train_files_f10


    test_files_f8 = load_files_from_folder(folder_f8_dyn, 'f8_dyn_')
    test_files_10 = load_files_from_folder(folder_f10_dyn, 'f10_dyn_')
    test_files = test_files_f8 + test_files_10

    train = [pd.read_csv(f, header=None) for f in train_files]
    train_data = pd.concat(train, ignore_index=True)

    test = [pd.read_csv(f, header=None) for f in test_files]
    test_data = pd.concat(test, ignore_index=True)

    x_y_train_input = train_data.iloc[:, :2]
    x_y_train_output = train_data.iloc[:, 2:]

    x_y_test_input = test_data.iloc[:, :2]
    x_y_test_output = test_data.iloc[:, 2:]

    input_scaler = MinMaxScaler()
    input_scaler.fit(x_y_train_input)

    output_scaler = MinMaxScaler()
    output_scaler.fit(x_y_train_output)

    x_y_input_train_scaled = input_scaler.transform(x_y_train_input)
    x_y_input_test_scaled = input_scaler.transform(x_y_test_input)

    x_y_output_train_scaled = output_scaler.transform(x_y_train_output)
    x_y_output_test_scaled = output_scaler.transform(x_y_test_output)

    np.save(os.path.join(output_folder_npy, 'X_train_scaled.npy'), x_y_input_train_scaled)
    np.save(os.path.join(output_folder_npy, 'x_y_train_output_scaled.npy'), x_y_output_train_scaled)
    np.save(os.path.join(output_folder_npy, 'X_test_scaled.npy'), x_y_input_test_scaled)
    np.save(os.path.join(output_folder_npy, 'x_y_test_output_scaled.npy'), x_y_output_test_scaled)

    pd.DataFrame(x_y_input_train_scaled).to_csv(os.path.join(output_folder_csv, 'X_train_scaled.csv'), index=False, header=False)
    x_y_train_output.to_csv(os.path.join(output_folder_csv, 'x_y_train_output.csv'), index=False, header=False)
    pd.DataFrame(x_y_input_test_scaled).to_csv(os.path.join(output_folder_csv, 'X_test_scaled.csv'), index=False, header=False)
    x_y_test_output.to_csv(os.path.join(output_folder_csv, 'x_y_test_output.csv'), index=False, header=False)

    np.save(os.path.join(output_folder_npy, 'x_y_train_input.npy'), x_y_train_input.values)
    np.save(os.path.join(output_folder_npy, 'x_y_test_input.npy'), x_y_test_input.values)

    np.save(os.path.join(output_folder_npy, 'x_y_train_output.npy'), x_y_train_output.values)
    np.save(os.path.join(output_folder_npy, 'x_y_test_output.npy'), x_y_test_output.values)

    return x_y_input_train_scaled, x_y_output_train_scaled, x_y_input_test_scaled, x_y_output_test_scaled


loading_all_files()