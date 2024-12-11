# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:27:01 2024

@author: TORTH
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
import re # this is used for string handling of polygon
import spherical_geometry.polygon as sp # Used for calculating area of polygon
import os
import glob
from tqdm import tqdm
import matplotlib.animation as animation
import time

def get_MET_ID(file_location, ID_type):
    string = ""
    with open(file_location, 'r') as file:
        string = file.read()
    rows = string.split("\n")
    i=0
    for row in rows:
        rows[i] = row.split(" ")
        i+=1
    # ID = np.asarray(re.split(r"[:\n]+", string), dtype=str)
    ID = np.asarray(rows)
    return ID[np.where(ID[:,0]==ID_type),1][0][0]

def httpGET_ualf_data_month(year, month, client_ID):
    """
    A function collecting lightning data for a given month in ualf format.
    (https://opendata.smhi.se/apidocs/lightning/parameters.html)

    Parameters
    ----------
    year : int
        The year we want data from.
    month : int
        The month we want data from.
    client_ID : string
        The client ID needed to extract data from met.no (Given when creating user at MET.no)

    Returns
    -------
    string
        A text string containing lightning data from desired period in the ualf format.

    """
    if(month < 12):
        reftime = str(year) + "-" + str(month) + "/" + str(year) + "-" + str(month+1)
    else:
        reftime = str(year) + "-" + str(month) + "/" + str(year+1) + "-" + str(1)
    endpoint = 'https://frost.met.no/lightning/v0.ualf?'
    parameters = {'referencetime': reftime}
    return requests.get(endpoint, parameters, auth=(client_id,'')).text # returns data as text string

def httpGET_ualf_data(client_ID, year_start, year_end, month_start=1, month_end=12):
    # header contains column parameters as defined by ualf format:
    header='d year month day hour min sec nanos lat lon pI multi nsens dof angle major minor chi2 rt ptz mrr cloud aI sI tI\n'
    year = year_start
    for year in range(year_start, year_end+1): # +1 as range does not include last int
        print(" ... Collecting data for ", year, "...")
        for month in tqdm(range(month_start, month_end+1)):
            header += httpGET_ualf_data_month(year, month, client_ID)
    return header

def ualf_to_array(ualf_string):
    ualf_rows = ualf_string.split("\n")
    i = 0
    for row in tqdm(ualf_rows):
        ualf_rows[i] = row.split(" ")
        i+=1
    return np.asarray(ualf_rows[:-1])

def array_to_ualf(ualf_array):
    ualf_string = ""
    print("Converting... Time for coffee? c(_)")
    for r in tqdm(range(ualf_array.shape[0])):
        for c in range(ualf_array.shape[1]):
            if c == 0:
                ualf_string += ualf_array[r, c]
            else:
                ualf_string += " " + ualf_array[r, c]
        ualf_string += '\n'
    return ualf_string

def save_ualf_data(ualf_data, filename):
    try:
        ualf_data.shape
        txt_file = array_to_ualf(ualf_data)
    except:
        txt_file = ualf_data
    with open(filename, "w") as txt_save:
        txt_save.write(txt_file)

def crop_ualf_data(polygon, ualf_data, only_ground_strikes=True):
    try:
        ualf_data.shape
        arr = ualf_data
    except:
        print("Converting string to array... This might take some time.")
        arr = ualf_to_array(ualf_data)
    
    polygon_numpy = np.asarray(re.split(r"[,\s]+", polygon), dtype=float)
    lons = polygon_numpy[[0,2,4,6]]
    lats = polygon_numpy[[1,3,5,7]]
    max_lat = max(lats)
    min_lat = min(lats)
    max_lon = max(lons)
    min_lon = min(lons)
    
    crop_list = []
    crop_list.append(arr[0,:])
    print("Now cropping dataset: ")
    for row in tqdm(arr[1:]):
        lat = float(row[8])
        lon = float(row[9])
        if lat < max_lat and lat > min_lat and lon < max_lon and lon > min_lon:
            if(only_ground_strikes):
                if row[-4].astype('int') == 0:
                    crop_list.append(row)
            else:
                crop_list.append(row)
    return np.asarray(crop_list)

def read_ualf(file_location):
    header = ""
    with open(file_location, 'r') as file:
        header = file.read()
    return header

def plot_lightning_history(ualf):
    try:
        ualf.shape
        arr = ualf
    except:
        arr = ualf_to_array(ualf)
    start_year = int(arr[1:][0,1])
    end_year = int(arr[1:][-1,1])
    start_month = 1#int(arr[1:][0,2])
    end_month = 12#int(arr[1:][-1,2])
    num_dates = (end_year-start_year + 1)*(end_month-start_month +1)
    
    dates = np.empty((num_dates),dtype=object)
    yr, mnth = 0, 0
    for d_i in range(dates.shape[0]):
        if start_month + mnth <= 9: # makes figure a little prettier:
            dates[d_i] = str(start_year + yr) + "/0" + str(start_month + mnth)
        else: 
            dates[d_i] = str(start_year + yr) + "/" + str(start_month + mnth)
        if (mnth > 10):
            yr += 1
            mnth = 0
        else:
            mnth += 1
    lightnings = np.zeros_like(dates, dtype=int)
    max_pI = np.zeros_like(dates, dtype=float)
    min_pI = np.zeros_like(dates, dtype=float)
    avg_pI = np.zeros_like(dates, dtype=float)
    
    date_i = 0
    sum_pI = 0.0
    for r_i, row in enumerate(arr[1:]): # We skip the first row as this contains string for column variables
        date_i = num_dates - (end_year - int(row[1]) + 1)*12 + (int(row[2]) - 1) # this gives index for current month in dates list
        dates[date_i] = row[1] + "/" + row[2]
        # avg_pI[date_i] = sum_pI/lightnings[date_i]
        lightnings[date_i] += 1 # Counting the lightning strike for given month
        # curr_pI = float(row[10])
        # sum_pI += curr_pI # Adding up pIs which will be averaged when going to next month
        if float(row[10]) > max_pI[date_i]:
            max_pI[date_i] = float(row[10])
        if float(row[10]) < min_pI[date_i]:
            min_pI[date_i] = float(row[10])
    
    fig, ax1 = plt.subplots()
    bar_color = 'black'
    max_pI_c = 'red'
    min_pI_c = 'orange'
    # Plotting list1 on the left y-axis
    bars = ax1.bar(dates, lightnings, color=bar_color)
    ax1.set_xlabel('Month')
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_ylabel('Lightning strikes [#]', color=bar_color)
    ax1.tick_params(axis='y')

    # Plot only non-zero min_pI values
    non_zero_min_pI_indices = [i for i in range(len(min_pI)) if min_pI[i] != 0]
    non_zero_min_pI_values = [np.abs(min_pI[i]) for i in non_zero_min_pI_indices]
    # Plot only non-zero max_pI values
    non_zero_max_pI_indices = [i for i in range(len(max_pI)) if max_pI[i] != 0]
    non_zero_max_pI_values = [max_pI[i] for i in non_zero_max_pI_indices]

    # Creating another y-axis for list2
    ax2 = ax1.twinx()
    offset=5
    # ax2.plot(max_pI, 'o', linestyle='', color=max_pI_c)
    ax2.plot(non_zero_max_pI_indices, non_zero_max_pI_values, 'o', linestyle='', color=max_pI_c)
    ax2.plot(non_zero_min_pI_indices, non_zero_min_pI_values, 'o', linestyle='', color=min_pI_c)
    for i in range(len(non_zero_max_pI_indices)):
        ax2.text(non_zero_max_pI_indices[i], non_zero_max_pI_values[i]+offset, float(non_zero_max_pI_values[i]), c=max_pI_c)
        ax2.text(non_zero_min_pI_indices[i], non_zero_min_pI_values[i]+offset, float(non_zero_min_pI_values[i]), c=min_pI_c)
    ax2.set_ylabel(f'Peak current [kA] (max = {max_pI_c}, min = {min_pI_c})', color='black')
    ax2.tick_params(axis='y', labelcolor=max_pI_c)

    i=0
    offset = 5
    for bar in bars:
        yval = bar.get_height()
        if yval != 0:
            ax1.text(bar.get_x(), yval, int(yval), va='bottom', c=bar_color)
        i+=1

    plt.title(f'Lightning data from {dates[0]} to {dates[-1]}')
    plt.show()
    
def plot_lightning_data(lightning_array, polygon):
    """
    Takes inn a numpy array of strings and plots the result. Also plots positions of Bergen, Sandnes and Oslo for reference positions

    Parameters
    ----------
    lightning_array : numpy array [str] (ualf format)
        A numpy array of strings in ualf format. [:,8] contains longitude and [:,9] contains latitude values

    Returns
    -------
    None.

    """
    try:
        lightning_array.shape
        lightning_array = lightning_array
    except:
        lightning_array = ualf_to_array(lightning_array)
    Bergen = np.array([60.24, 5.2])
    Sandnes = np.array([58.5, 5.43])
    Oslo = np.array([59.54, 10.45])

    poly_request = polygon.split(",")
    c1 = poly_request[0].split(" ")
    c2 = poly_request[1].split(" ")
    c3 = poly_request[2].split(" ")
    c4 = poly_request[3].split(" ")
    # Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # get the years we have data from:
    years = np.unique(lightning_array[1:, 1].astype(int))
    # We then need unique color for each year:
    num_colors = years[-1]-years[0]+1
    # Create unique colors from viridis color scheme:
    colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
    # And create a dictionary to collect color by year:
    year_to_color = {year: colors[i] for i, year in enumerate(years)}
    
    for row in lightning_array[1:]:
        plt.scatter(row[9].astype('float'), row[8].astype('float'), color=year_to_color[row[1].astype('int')])
    # plt.scatter(lightning_array[1:,9].astype('float'), lightning_array[1:,8].astype('float'))
    # plt.scatter(Bergen[1], Bergen[0], label='Bergen', c='orange')
    # plt.scatter(Sandnes[1], Sandnes[0], label='Sandnes', c='g')
    # plt.scatter(Oslo[1], Oslo[0], label='Oslo', c='y')
    plt.scatter(float(c1[0]), float(c1[1]), label='c1', c='r')
    plt.scatter(float(c2[0]), float(c2[1]), label='c2', c='r')
    plt.scatter(float(c3[0]), float(c3[1]), label='c3', c='r')
    plt.scatter(float(c4[0]), float(c4[1]), label='c4', c='r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # Add legend for years
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=year_to_color[year], markersize=10, label=str(year)) for year in years]
    plt.legend(handles=handles, title="Year")
    ax.set_aspect('equal')
    
    N = lightning_array[1:].shape[0]/square_area(polygon)/(np.abs(years[-1]-years[0])+1)
    
    plt.title(f'{lightning_array.shape[0]} lightnings occured from {lightning_array[1,1]} to {lightning_array[-1,1]} within region. N = {N:.3f}')
    plt.show()

def square_area(polygon):
    """
    Takes in a string with polygon corners and calculates the area

    Parameters
    ----------
    polygon : string
        A text string with polygon corners.

    Returns
    -------
    float
        The area in km2 within polygon.

    """
    polygon_numpy = np.asarray(re.split(r"[,\s]+", polygon), dtype=float)
    lons = polygon_numpy[[0,2,4,6]]
    lats = polygon_numpy[[1,3,5,7]]

    # Create a spherical polygon
    polygon2 = sp.SphericalPolygon.from_lonlat([lons[0], lons[1], lons[2], lons[3]], [lats[0], lats[1], lats[2], lats[3]])
    return polygon2.area()*6378000**2*10**-6 # area formula returns steradians so I multiply with earth radius (A=4piR^2)

def get_N(polygon, txt_file):
    """
    Calculates the lightning density N [#lightning/km2/year] for the region within the polygon

    Parameters
    ----------
    polygon : string
        A text string with polygon corners..
    txt_file : string
        location of text file to read data from (ualf format).

    Returns
    -------
    float
        The value of N [#lightning/km2/year].

    """
    ualf = read_ualf(txt_file)
    array = ualf_to_array(ualf)
    yrs = np.unique(array[1:,1].astype('int'))
    yr_span = np.abs(yrs[-1]-yrs[0])+1
    area = square_area(polygon)
    return array.shape[0]/area/yr_span

def get_year_array(ualf, year):
    try:
        ualf.shape
        arr = ualf
    except:
        arr = ualf_to_array(ualf)
        
    year_arr = arr[np.where(arr[1:,1].astype('int') == year)]
    return year_arr

######################################################################################################
# !!! Insert the location you saved the python file !!!:
code_loc = r'C:\Users\TORTH\OneDrive - Lyse AS\Rotasjon 5 - Netteknikk\Lynnedslag\Code_test'
os.chdir(code_loc)
os.getcwd() # This prints the location your files will be saved
######################################################################################################

##### Setup collection parameters for lightning data from frost.met.no: ##############################
client_id = get_MET_ID(code_loc+'\MET_IDs', 'client_id')    # This is given when creating a user at met.no (https://frost.met.no/howto.html)
client_secret = get_MET_ID(code_loc+'\MET_IDs', 'client_secret')
reftime='2011-05/2011-06'                           # Here we give the period which we want to collect, can be no larger than 1 month (yyyy-mm-dd/yyy-mm-dd)
maxage='P1D'                                        # Not sure what this is used for...

# This is the polygon which we are interested in (This does not work in GET request...):
polygon = '6.14 59.1,6.40 59.10,6.40 59.0,6.14 59.0'

# We set an endpoint which is the webpage we want to collect data from
endpoint = 'https://frost.met.no/lightning/v0.ualf?' 
# We add our collection parameters to the enpoint URL and issue an HTTP GET request:
parameters = {
    'referencetime': reftime,
    'polygon': polygon,
    'maxage': maxage,
}

# A small startup test:
r = requests.get(endpoint, parameters, auth=(client_id,''))
# Can now print the data collected:
r.text

######################################################################################################

# We collect lightning data from start year to end year:
# ualf_string = httpGET_ualf_data(client_id, year_start=2014, year_end=2024, month_start=1, month_end=12)
ualf_string = read_ualf(code_loc+'\HTTPGET_2014_2024.txt')

# Then crop out only relevant data for our region of interest:
cropped = crop_ualf_data(polygon, ualf_string, only_ground_strikes=False)
cropped_gs = crop_ualf_data(polygon, cropped, only_ground_strikes=True)

# Save our data of interest to text file (In folder set at top of code as code_loc):
save_ualf_data(cropped, "HTTPGET_2014_2024_crop.txt")
save_ualf_data(cropped_gs, "HTTPGET_2014_2024_crop_groundStrikes.txt")

# If saved from before we can read it from text file:
ualf_string_crop = read_ualf(code_loc+'\HTTPGET_2014_2024_crop.txt')
ualf_string_crop_gs = read_ualf(code_loc+'\HTTPGET_2014_2024_crop_groundStrikes.txt')

# Can convert ualf string data to array:
cropped = ualf_to_array(ualf_string_crop)
cropped_gs = ualf_to_array(ualf_string_crop_gs)

# Now we can plot history of lighting strikes per month for timespan (if there are noe lightnings the first year collected, this year will not show up)
plot_lightning_history(cropped)
plot_lightning_history(ualf_string_crop_gs)

# And finally we can plot the location of the lightning strikes and get a feel for the amount and placement.
plot_lightning_data(cropped, polygon)
plot_lightning_data(cropped_gs, polygon)

# The calculated value for N_g is now:
N = get_N(polygon, code_loc+'\HTTPGET_2014_2024_crop.txt')
print(N)

# Animation of lightnings per year:
arr = cropped_gs
years = np.unique(arr[1:,1])

# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o')

# Define axis limits:
polygon_numpy = np.asarray(re.split(r"[,\s]+", polygon), dtype=float)
lons = polygon_numpy[[0,2,4,6]]
lats = polygon_numpy[[1,3,5,7]]
max_lat = max(lats)
min_lat = min(lats)
max_lon = max(lons)
min_lon = min(lons)
offset = 0.02
ax.set_xlim(min_lat-offset, max_lat+offset)
ax.set_ylim(min_lon-offset, max_lon+offset)

# We then need unique color for each year:
num_colors = years[-1].astype('int')-years[0].astype('int')+1
# Create unique colors from viridis color scheme:
colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
# And create a dictionary to collect color by year:
year_to_color = {year: colors[i] for i, year in enumerate(years)}

# Initialize the line
def init():
    line.set_data([], [])
    return line,

def update(frame):
    curr_year = arr[1,1].astype('int')+frame
    frame_arr = get_year_array(arr, curr_year)[1:] # removing header line
    x = frame_arr[:,8].astype('float')
    y = frame_arr[:,9].astype('float')
    line.set_data(x, y)
    line.set_color(year_to_color[str(arr[1,1].astype('int')+frame)])
    time.sleep(0.5)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(years), init_func=init, blit=True)
plt.show()

