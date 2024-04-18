from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def create_sequences(data, FEATURE_COLUMNS, window_size, number):
    sequences = []
    labels = []
    noises = []
    np.random.seed(10)
    for disk_id, group in tqdm(data.groupby("disk_id")):
        if group['label'].sum() > 0:
            group = group.iloc[-(number+window_size):,:]
            group.iloc[-number:,-2] = 1
        if group['noise'].sum() > 0:
            group.iloc[:,-1] = 0
            group.iloc[-number:,-1] = 1

        if group['label'].sum() > 0:
            for i in range(group.shape[0]-window_size):
                sequence_features = group.iloc[i:i + window_size][FEATURE_COLUMNS]     
                sequence_features = sequence_features.to_numpy().transpose(1,0)
                label = group.iloc[i+window_size, -2]
                noise = group.iloc[i+window_size, -1]
                
                sequences.append(sequence_features)
                labels.append(label)
                noises.append(noise)

        else:
            if group.shape[0] >= 121:
                start_list = np.random.randint(0, group.shape[0]-window_size, 30).tolist()
                for i in start_list:
                    sequence_features = group.iloc[i:i + window_size][FEATURE_COLUMNS]
                    sequence_features = sequence_features.to_numpy().transpose(1,0)      
                    label = group.iloc[i+window_size, -2]
                    noise = group.iloc[i+window_size, -1]
                    
                    sequences.append(sequence_features)
                    labels.append(label)
                    noises.append(noise)

    return np.array(sequences), np.array(labels), np.array(noises)

def load_ssd_dataset(args):
    # 데이터 로딩 및 전처리 로직
    ## 15:1 상황
    noise = int(args.r*100)

    train_df = pd.read_csv(f'/home/iai3/Desktop/Bongjun/SSD_Failure/dataset_noised/prev_next_1/noise_{noise}_percent/noised_{noise}_percent_train.csv')
    val_df = pd.read_csv(f'/home/iai3/Desktop/Bongjun/SSD_Failure/dataset_noised/prev_next_1/noise_{noise}_percent/noised_{noise}_percent_validation.csv')
    
    #train, val 합쳐서 train_df로 사용함.(validation 없음)
    train_df = pd.concat([train_df,val_df])
    test_df = pd.read_csv('/home/iai3/Desktop/Bongjun/SSD_Failure/dataset_noised/clean_test.csv')

    train_df = reduce_mem_usage(train_df, verbose=True)
    test_df = reduce_mem_usage(test_df, verbose=True)

    # for debugging
    # train_df = train_df[:100000]
    # test_df = test_df[:100000]

    train_df.rename(columns={'noise_injected':'noise'}, inplace=True)

    #final_X_train = final_train_set.drop(columns=['disk_id','ds','label'])
    X_train = train_df.drop(columns=['disk_id','ds','label', 'noise'])
    X_test = test_df.drop(columns=['disk_id','ds','label'])

    #final_y_train = final_train_set['label']
    y_train = train_df['label']
    y_test = test_df['label']

    y_train_noise = train_df['noise']
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled_df = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    del X_train, X_test

    train_scaled = pd.concat([train_df[['disk_id','ds']],X_train_scaled_df, y_train, y_train_noise], axis=1)
    test_scaled = pd.concat([test_df[['disk_id','ds']],X_test_scaled_df, y_test], axis=1)
    test_scaled.loc[:, 'noise'] = 0
    del y_train, y_test, y_train_noise

    FEATURE_COLUMNS = train_df.columns[2:-2]
    FEATURE_COLUMNS = ['disk_id'] + list(FEATURE_COLUMNS)

    window_size = 90 
    number = 30
    train_data, noise_label, train_noise_mask = create_sequences(train_scaled, FEATURE_COLUMNS, window_size, number)
    data, test_label, _ = create_sequences(test_scaled, FEATURE_COLUMNS, window_size, number)
    train_label = np.where(train_noise_mask == 1, 1 - noise_label, noise_label)
    
    del train_scaled
    
    print('Num of train :', train_data.shape[0])
    print('Num of test :', data.shape[0])
    # test_scaled 는 마지막 final_test에서 필요해서 return해주는거임!
    return train_data, train_label, noise_label, data, test_label, test_scaled


class ssd_dataset(Dataset):
    def __init__(self, data, real_label, label, mode):

        self.mode = mode
        self.data = data
        self.label = label
        self.mode = mode
        #self.probability = None
        self.real_label = real_label

        if self.mode == 'eval':
            self.noise_mask = real_label != label

    def __getitem__(self, index):
        if self.mode == 'train':
            seq, target = self.data[index], self.label[index]
            return seq, target, index
        
        elif self.mode == 'eval_train':
            seq, target, noise_mask = self.data[index], self.label[index], self.noise_mask[index]
            return seq, target, index, noise_mask

        elif self.mode == 'test':
            seq, target = self.data[index], self.label[index]
            return seq, target

    def __len__(self):
            return len(self.data)


class ssd_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, args, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.args = args

        self.train_data, self.train_label, self.noise_label, self.test_data, self.test_label, self.test_scaled = load_ssd_dataset(self.args)

    def run(self, mode):
        if mode == 'train':
            train_dataset = ssd_dataset(data=self.train_data, real_label= self.noise_label, label= self.train_label, mode="train")
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader,np.asarray(self.noise_label),np.asarray(self.train_label)

        elif mode == 'test':
            test_dataset = ssd_dataset(data= self.test_data, real_label= None, label= self.test_label, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        
        elif mode == 'eval_train':
            eval_dataset = ssd_dataset(data=self.dataset, mode='train')
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader,np.asarray(self.noise_label),np.asarray(self.train_label)
        
        elif mode == 'final_test':
            test_scaled = self.test_scaled
            return test_scaled