import numpy as np
from scipy import signal
import os
from scipy.stats import kurtosis
from scipy.io import loadmat

############ NOTE: ############
#  The function at the bottom: 
#  preprocess_voltage
#  is the one you want to use and pay attention to 
#  and the load_volt_data
#  this one loads in all brains
#  and the load_loc  #! DO NOT USE, STILL UNDER CONSTRUCTION
#  this one loads in all electrode locations (in native brain space)


#! NOTE: FOR SOME REASON THIS FUNCTION DOESNT LIKE TO IMPORT TO ANOTHER FILE
#! BUT IT STILL WORKS IF YOU JUST COPY IT IN
# give it the full directory path to the file with all the brains 
def loadvoltdata(brains_volt_path): 
    #column_names = ['aa', 'ap', 'ca', 'de', 'fp', 'ha', 'ja', 'jm', 'jt', 'mv','rn','rr','wc','zt']
    raw_volt_data = []
    raw_stim_data = []
    with os.scandir(brains_volt_path) as entries:
        for entry in entries: #for each brain
            root, extension = os.path.splitext(entry.path) #to avoid any addition hidden files
            if entry.is_file() and extension =='.mat':
                volt_data = loadmat(entry.path)
                raw_volt_data.append(volt_data['data'])
                raw_stim_data.append(volt_data['stim'])
    return raw_volt_data, raw_stim_data



######## BUTTERWORTH NOTCH FILTER FUNCTION: ########
# using bandstop filter (allowing freqiuences higher and lower to pass)
# using a digital filter (hence the analog param)
# sos output (as scipy recommends for general filtering)
# removing freq between the lowcut and highcut amounts (set to match with the samplerate fs)
# returns a filter to be used on the data
# this filter is actually the coefficents to a function that is applyed to the frequency data
# this function gets called into use in the 'sosfiltfilt' function used later
def butter_notch(lowcut, highcut, fs, order=4):
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], btype='bandstop', analog=False, output='sos', fs=fs)
    return sos

# takes the voltage data, the low and high ends of what freq you want to cut out, the sample rate and order of butter worth filter
# applyes a filter function to the data using coefficenets that were calcuated by the above function
def butter_notch_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_notch(lowcut, highcut, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y

####### Applying the butterworth function: #######
# Give it the raw voltage data, and optionally what freq you want to remove (defaults to [60,120,180,240])
# and optionally give it the range around that freq you also want gone (defaults .5)
def apply_butter_filter(raw_volt_data,bad_freq= [60,120,180,240],width=.5):
    butter_filtered_volt = []
    for voltage in raw_volt_data:
        filtered_brain = []
        for node in range(voltage.shape[1]): #for each node
            node_volt = voltage[:,node] #gets that nodes voltage data
            #filtered_data = butter_notch_filter(node_volt, 59.5, 60.5, 1000, order=4) #filters out the only 60hz range 
            #''' #use to run several fequencies and not just the 60 one
            filtered_data = node_volt
            for freq in bad_freq: #applyes the butter worth function for each frequency you want removed
                node_volt = butter_notch_filter(node_volt, freq-width, freq+width, 1000, order=4) #filters out the 60hz range 
            #'''
            filtered_brain.append(filtered_data)

        filtered_brain = np.array(filtered_brain)
        butter_filtered_volt.append(filtered_brain) #adds the filted brain to the list    
    return butter_filtered_volt
# NOTE: 
# Now the dimensions have flipped in the output of this function, before it was (brain,timestep,electrode)
# now it is (brain,electrode,timestep)
#

###### Kurtosis function: ######
# give it the butterworth filtered voltage
# returns all the electrodes that pass the kurtosis test of a score less then 10
# this allows use to filter out any potentially epilictic brain activity
def apply_kurtosis_check(butter_filtered_volt):
    kurtosis_filtered_v = []
    for brain in butter_filtered_volt:
        temp_node = []
        for node in brain:
            k_score = kurtosis(node) #calcuates the kurtosis score for the elctrode data of this node and brain
            if k_score < 10: #only keep if its score is less then 10
                temp_node.append(node)
        temp_node = np.array(temp_node)
        kurtosis_filtered_v.append(temp_node)
    return kurtosis_filtered_v

#simply check that makes sure each brain as at least two electrodes
def ensure_node_count(kurtosis_filtered_v):
    processed_data = []
    for brain in kurtosis_filtered_v:
        if brain.shape[0] > 2:
            processed_data.append(brain)
    return processed_data

###### Preprocess function: ######
# this applys all the preprocessing functions desribed above 
# applys the butterwroth notch
# applys the kurtosis check to remove odd behaivor
# removes brains with less then 2 nodes
def preprocess_voltage(raw_voltage):
    butter_filted_volt = apply_butter_filter(raw_voltage) #using default settings
    kurtosis_filtered_volt = apply_kurtosis_check(butter_filted_volt)
    node_filtered_volt = ensure_node_count(kurtosis_filtered_volt)
    return node_filtered_volt




#! DO NOT USE, STILL UNDER CONSTRUCTION
def load_loc(directory_path):
    master_list = []
    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_file():
                try:
                    electrodes = loadmat(entry.path)
                    locs = electrodes['locs']
                    for point in locs:
                        master_list.append(point)
                except:
                    print("some other file type was found")
    return master_list