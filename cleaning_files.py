from preprocessing import trim_ends
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd



path_to_dataset_accelerometer='/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/motion'
path_to_dataset_ibi='/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/heart_rate'
path_to_labels = '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/labels'

path_to_acc_files=[file for file in glob.glob(path_to_dataset_accelerometer+'/**/*.txt', recursive=True)]
path_to_ibi_files=[file for file in glob.glob(path_to_dataset_ibi+'/**/*.txt', recursive=True)]

participant_id=[file.split('/')[10].split('_')[0] for file in glob.glob(path_to_dataset_accelerometer+'/**/*.txt', recursive=True) ]
scaler = StandardScaler()
directory = './results'
if not os.path.exists(directory):
    os.makedirs(directory)
    
dataset = pd.DataFrame()

for participant in participant_id:
    #iterating through all the participants in the dataset
    #reading the acceleration files
    print(f"Working on Particpant: {participant}")
    path_to_acc = '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/motion/'+participant+'_acceleration.txt'
    participant_acc=pd.read_csv(path_to_acc,sep=' ')
    participant_acc.columns=['time_to_sleep','ax','ay','az'] #renaming the columns
    participant_acc=trim_ends(participant_acc) #just getting the sleep times 
    path_to_ibi= '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/heart_rate/'+participant+'_heartrate.txt' 
    participant_ibi=pd.read_csv(path_to_ibi,sep=',') #reading the ibi values
    participant_ibi.columns = ['time_to_sleep','ibi'] #renaming the columns
    participant_ibi=trim_ends(participant_ibi) #getting only the sleep times
    #merging these files 
    participant_ibi["marker"] = "ibi"
    participant_ibi = participant_ibi.rename(columns={'ibi': "value"})
    
    df_accx=participant_acc[['time_to_sleep', 'ax']]
    df_accx["marker"] = "ax"
    df_accx = df_accx.rename(columns={'ax': "value"})
    df_participant =participant_ibi.append(df_accx)
    df_accy=participant_acc[['time_to_sleep', 'ay']]
    df_accy["marker"] = "ay"
    df_accy = df_accy.rename(columns={'ay': "value"})
    df_participant =df_participant.append(df_accy)
    df_accz=participant_acc[['time_to_sleep', 'az']]
    df_accz["marker"] = "az"
    df_accz = df_accz.rename(columns={'az': "value"})
    df_participant =df_participant.append(df_accz)
    df_participant["participant"]= participant
    print(df_participant)
    df_sensor_pivot =df_participant.pivot_table(index=['time_to_sleep','participant'], columns=['marker'], values=["value"])
    
    df_sensor_pivot.columns= ['ax','ay','az','ibi']
    df_sensor_pivot= df_sensor_pivot.ffill().bfill()
    df_sensor_pivot = df_sensor_pivot.reset_index()
    
        
    path_to_labels = '/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/sleep-accel/1.0.0/labels/'+ participant+'_labeled_sleep.txt'
    participant_labels=pd.read_csv(path_to_labels,sep=' ') # Reading the labels of sleep stages
    participant_labels.columns=['time_to_sleep','sleep_stage'] #sleep stages are only given for times >0.
    participant_labels['time_to_sleep'] = participant_labels['time_to_sleep'].astype('float64') #making it float64 to merge 
    
    df_sensor_pivot = df_sensor_pivot.merge(participant_labels, on='time_to_sleep', how='outer')
    df_sensor_pivot= df_sensor_pivot.ffill().bfill()
    print(df_sensor_pivot)
    

    """
    plt.plot(merged_dataframe['time_to_sleep'],merged_dataframe['ax'],label='ax')
    plt.plot(merged_dataframe['time_to_sleep'],merged_dataframe['ay'],label='ay')
    plt.plot(merged_dataframe['time_to_sleep'],merged_dataframe['az'],label='az')
    plt.plot(merged_dataframe['time_to_sleep'],merged_dataframe['ibi'],label='ibi')
    plt.plot(merged_dataframe['time_to_sleep'],merged_dataframe['sleep_stage'],label='sleep_stage')
    plt.xlabel('time_asleep(s)')
    plt.ylabel('a in g)')
    plt.legend()
    #plt.show()
    plt.savefig('./results/'+participant+'.png')
    """
    
    epoch_length= 30
    end_index = int(df_sensor_pivot.tail(1)['time_to_sleep'])
    
    num_epochs=df_sensor_pivot.tail(1)['time_to_sleep']//30
    print(f'number of sleep epochs: {num_epochs}')
    for i in range(30,end_index,epoch_length):
        next_thirty_seconds = df_sensor_pivot[(df_sensor_pivot['time_to_sleep'] > i) & (df_sensor_pivot['time_to_sleep'] <= i+30)]
        #print(f'length of the window = {len(next_thirty_seconds)}')
        if len(next_thirty_seconds) >= 850:
            next_thirty_seconds = next_thirty_seconds.sample(850)
            dataset = dataset.append(next_thirty_seconds)
        
        
        else:
            print("Discarding window starting at index:", i)
            print(f'length of the window = {len(next_thirty_seconds)}')
            
        

dataset.to_csv('/nesi/nobackup/aut03802/dataset_sleep/physionet.org/files/data_compiled.csv')
                    
                    
    
          
    
    
    
    
    




