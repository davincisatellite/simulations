import pandas as pd
import os
import urllib.request
from datetime import datetime, timedelta, date
import numpy as np
from tudatpy.kernel import constants


J2000_to_unix_timestamp_delta_t = 946684800 # https://stackoverflow.com/questions/35763357/conversion-from-unix-time-to-timestamp-starting-in-january-1-2000

folder_root = os.path.dirname(os.path.realpath(__file__))

def extract_values_from_csv(filepath):
    """will read a csv file formatted with the columns: label,value,units,description
    the string under "item" is used as the key of the resulting dictionary and the 
    corresponing value is associated with it.

    Args:
        filepath (string): path of csv file 

    Returns:
        dict: dictionary containing values stored in the csv
    """
    
    df = pd.read_csv(filepath)

    values = {}
    # reading rows of excel file
    for _, row in df.iterrows():
        try:
            values[row.label] = float(row.value)
        except ValueError:
            values[row.label] = row.value
    return values


def gen_csv_from_nrlmsise_data(path):
    
    data = []
    
    def cast_to_integer(str):
        try:
            return int(str)
        except ValueError:
            return np.nan
        
    def cast_to_float(str):
        try:
            return float(str)
        except ValueError:
            return np.nan
        
    
    with open(path) as f:
        for line in f:
            # check if row contains data or text
            new_row = {}
            if line[0] in ["1", "2"]:
                # row contains data
                new_row["year"] = cast_to_integer(line[0:4])
                new_row["month"] = cast_to_integer(line[5:7])
                new_row["day"] = cast_to_integer(line[8:10])
                new_row["BSRN"] = cast_to_integer(line[11:15])
                new_row["ND"] = cast_to_integer(line[16:18])
                new_row["Kp_1"] = cast_to_integer(line[19:21])
                new_row["Kp_2"] = cast_to_integer(line[22:24])
                new_row["Kp_3"] = cast_to_integer(line[25:27])
                new_row["Kp_4"] = cast_to_integer(line[28:30])
                new_row["Kp_5"] = cast_to_integer(line[31:33])
                new_row["Kp_6"] = cast_to_integer(line[34:36])
                new_row["Kp_7"] = cast_to_integer(line[37:39])
                new_row["Kp_8"] = cast_to_integer(line[40:42])
                new_row["Kp_sum"] = cast_to_integer(line[43:46])
                new_row["Ap_1"] = cast_to_integer(line[47:50])
                new_row["Ap_2"] = cast_to_integer(line[51:54])
                new_row["Ap_3"] = cast_to_integer(line[55:58])
                new_row["Ap_4"] = cast_to_integer(line[59:62])
                new_row["Ap_5"] = cast_to_integer(line[63:66])
                new_row["Ap_6"] = cast_to_integer(line[67:70])
                new_row["Ap_7"] = cast_to_integer(line[71:74])
                new_row["Ap_8"] = cast_to_integer(line[75:78])
                new_row["Ap_avg"] = cast_to_integer(line[79:82])
                new_row["Cp"] = cast_to_float(line[83:86])
                new_row["C9"] = cast_to_integer(line[87:88])
                new_row["ISN"] = cast_to_integer(line[89:92])
                new_row["Adj_F107"] = cast_to_float(line[93:98])
                new_row["Q"] = cast_to_integer(line[99:100])
                new_row["Adj_Ctr81"] = cast_to_float(line[101:106])
                new_row["Adj_Lst81"] = cast_to_float(line[107:112])
                new_row["Obs_F107"] = cast_to_float(line[113:118])
                new_row["Obs_Ctr81"] = cast_to_float(line[119:124])
                new_row["Obs_Lst81"] = cast_to_float(line[125:130])
                
                data.append(new_row)
                
                
    df = pd.DataFrame.from_records(data)
    
    csv_filepath = os.path.splitext(path)[0]+ ".csv"
    
    df.to_csv(csv_filepath, index=False)      


def get_space_weather_data_txt_file_path(force_update=False):
    """will download space weather datafile from clestrack.org if current one is older than 7 weeks.

    Args:
        force_update (bool, optional): force update of space weather file?. Defaults to False.

    Returns:
        (string): path to space weather datafile.
    """

    # set the URL of the file you want to download
    url = "https://celestrak.org/SpaceData/sw19571001.txt"

    # set the path of the local file
    local_file_path = "data/celestrak_space_weather_file_for_atmosphere_model.txt"

    # set the time threshold for the file to be considered old (1 week in this example)
    time_threshold = datetime.now() - timedelta(days=1)

    # check if the file exists and is older than the time threshold
    if os.path.exists(local_file_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(local_file_path))
        if not mod_time > time_threshold:
            # if file is old than the time threshold, download it
            force_update = True
    else:
        force_update = True
            
    if force_update:
        # if the local file is not up to date, download the new file
        with urllib.request.urlopen(url) as response:
            with open(local_file_path, 'wb') as file:
                file.write(response.read())
                
        gen_csv_from_nrlmsise_data(local_file_path)
        # print('New file downloaded.')
    return local_file_path


def get_space_weather_data_csv_file_path(force_update=False):
    txt_file_path = get_space_weather_data_txt_file_path(force_update)
    
    csv_file_path = os.path.splitext(txt_file_path)[0]+ ".csv"
    
    return csv_file_path
    

def load_nrlmsis00_arrays():
    
    df = pd.read_csv(get_space_weather_data_csv_file_path())
    
    years = df["year"].to_numpy()
    months = df["month"].to_numpy()
    days = df["day"].to_numpy()
    Ap_1 = df["Ap_1"].to_numpy()
    Ap_2 = df["Ap_2"].to_numpy()
    Ap_3 = df["Ap_3"].to_numpy()
    Ap_4 = df["Ap_4"].to_numpy()
    Ap_5 = df["Ap_5"].to_numpy()
    Ap_6 = df["Ap_6"].to_numpy()
    Ap_7 = df["Ap_7"].to_numpy()
    Ap_8 = df["Ap_8"].to_numpy()
    f107_obs = df["Obs_F107"].to_numpy()
    f107_obs_ctr81d = df["Obs_Ctr81"].to_numpy()
    
    dates = np.array([date(y, m, d) for y, m, d in zip(years, months, days)])
    
    Aps = np.array([Ap_1, Ap_2, Ap_3, Ap_4, Ap_5, Ap_6, Ap_7, Ap_8]).T
    
    return dates, Aps, f107_obs, f107_obs_ctr81d

def asd():
    print(nrlmsis00_dates)

# get_space_weather_data_txt_file_path()
nrlmsis00_dates, nrlmsis00_Aps, nrlmsis00_f107_obs, nrlmsis00_f107_obs_81d = load_nrlmsis00_arrays()

if __name__ == "__main__":
    
    get_space_weather_data_txt_file_path()

    panel_surface_normals = np.zeros((6,3))              # normals of each panel surfaces in body-fixed reference frame
        
    print(panel_surface_normals)
    
    
    DVS = satellite("dvs_properties.csv")
    
    print(dir(DVS))
