# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:54:26 2024

@author: TORTH
"""

import urllib
from PIL import Image
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import hyperspy.api as hs
import requests
import re # this is used for string handling of polygon
import spherical_geometry.polygon as sp # Used for calculating area of polygon
import os
import glob
from tqdm import tqdm
import time
import matplotlib.animation as animation

# Insert the location you saved the python file:
code_loc = r'C:\Users\TORTH\OneDrive - Lyse AS\Rotasjon 5 - Netteknikk\Lynnedslag\iteration_4'
os.chdir(code_loc)
os.getcwd() # This prints the location your files will be saved

def getMETid(file_location, ID_type):
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

class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """
    
    _lat = None
    _lon = None
    _zoom = None
    hs_im = None
    rgb_im = None
    
    def __init__(self, lat=59.3, lng=5.0, zoom=10):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required, defaults to 59.3
                lng:    The longitude of the location required, defaults to 5.0
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 10
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude 
            and zoom level
            Returns:    An X,Y tile coordinate
        """
        
        tile_size = 256 # Google base map has 256x256 pixels for entire world
        
        # For a given zoom level we get 2**zoom tiles
        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # We acquire the tile coordinates at given zoom level for top left corner of our map
        # Find the x_point given the longitude
        point_x = (tile_size/ 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1+sin_y)/(1-sin_y)) * -(tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.
            
            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None :
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        #Create a new image of the size require
        map_img = Image.new('RGB', (width,height))

        for x in range(0, tile_width):
            for y in range(0, tile_height) :
                url = 'https://mt0.google.com/vt/lyrs=y&?x='+str(start_x+x)+'&y='+str(start_y+y)+'&z='+str(self._zoom) # lyrs collects photo imagery
                    
                current_tile = str(x)+'-'+str(y)
                urllib.request.urlretrieve(url, current_tile)
            
                im = Image.open(current_tile)
                map_img.paste(im, (x*256, y*256))
              
                os.remove(current_tile)
        self.rgb_im = map_img
        
        # Create a hyperspy image of the collected data for roi handling (pixel_coords=tile_coords*256 as each tile has 256x256 resolution):
        img_data = np.asarray(map_img.convert('L'))
        im = hs.signals.Signal2D(img_data)
        im.axes_manager[0].name = "Longitude"
        im.axes_manager[1].name = "Latitude"
        im.axes_manager["Longitude"].scale = 1
        im.axes_manager["Latitude"].scale = 1
        im.axes_manager["Longitude"].units = "px East"
        im.axes_manager["Latitude"].units = "px North"
        x_tile, y_tile = self.getXY() # This collects tile coords upper left corner (Not the same as passed lat/lon coordinates which is slightly inside tile!)
        im.axes_manager["Longitude"].offset = x_tile*256 # Must be done this way to capture offset due to tile cut-out from maps (not identical to pixel for lat lon which is of higher resolution)
        im.axes_manager["Latitude"].offset = y_tile*256
        self.hs_im = im

    def getImageHS(self):
        return self.hs_im
    
    def getImageRGB(self):
        return self.rgb_im
    
class METuartDownloader:
    """
        A class handling uart data collected from MET url: 'https://frost.met.no/lightning/v0.ualf?' 
        Given longitude, latitude and zoom level google map images are downloaded and displayed with the lightning data.
    """
    
    im = None
    
    # These are used for upperleft conrner of overview map
    lat_ov = None
    lon_ov = None
    zoom_ov = None # Default value in init is 10 for a large area
    gmd_ov = None # object as defined above, used to download overview map
    map_ov_hs = None # Keep one hs image for handling roi later
    
    # These are used for upperleft conrner of roi map
    lat_roi = None
    lon_roi = None
    zoom_roi = None # Default value is 12 for area of interest, this might be a bit too high for large areas
    gmd_roi = None # object as defined above, used to download area of interest map
    map_roi_rgb = None # PIL Image, This will be used to display results on top of
    polygon = None # roi of interest in string format as required by MET url (This does not actually filter on the roi)
    roi_xTiles = None # Number of "Google tiles" we need to span roi (https://developers.google.com/maps/documentation/javascript/coordinates)
    roi_yTiles = None
    
    # Used for keeping track of corners of rgb map image (Which is deliberatley a little larger than the roi/polygon)
    roi_start_lat = None
    roi_start_lon = None
    roi_end_lat = None
    roi_end_lon = None
    
    # Hyperspy variables used for roi determination
    roi = None
    im_roi = None
    
    # MET url collection variables. client_id is given when creating a user at MET (https://frost.met.no/howto.html)
    client_id = None
    client_secret = None
    reftime = None
    Maxage = None
    enpoint = 'https://frost.met.no/lightning/v0.ualf?' 
    
    # Extracted data from MET (ualf is a universal format for lightning data: https://opendata.smhi.se/apidocs/lightning/parameters.html)
    ualf = None
    ualf_crop = None # This is an array in current code
    
    # Messy init code for client_id, leave for now but fix later...
    def __init__(self, lat=59.3, lng=5.0, zoom_overview=10):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required, defaults to 59.3
                lng:    The longitude of the location required, defaults to 5.0
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 10
        """
        try:
            self.client_id = getMETid(code_loc+'\MET_IDs', 'client_id')
        except:
            self.client_id = input("Attempt at reading client ID from file in cwd was unsuccessful, please input client_id given by MET manually here: ")
        self.lat_ov = lat
        self.lng_ov = lng
        self.zoom_ov = zoom_overview
        self.gmd_ov = GoogleMapDownloader(lat=lat, lng=lng, zoom=zoom_overview) # This creates a gmd object containing hs image and rgb image of overview area (defaults to lat 59.3, lon 5.0)
        self.gmd_ov.generateImage()
        self.map_ov_hs = self.gmd_ov.getImageHS()
        
    def pixel_to_lat_lon(self, x_pixel, y_pixel, zoom=0):
        """
        Parameters
        ----------
        x_pixel : int
            Pixel on world map at given zoom level.
        y_pixel : int
            Pixel in wolrd map at given zoom level.
        zoom : int
            Zoom level at which the input pixel values are corresponding to. Needed for correct convertion from pixel to lat/lon space. The default is 0.

        Returns
        -------
        lat : float
            Latitude.
        lon : float
            longitude.

        """
        # At a given zoom level the world is divided into 256*2**zoom_level pixels
        map_size=256*2**zoom
        # Normalize pixel coordinates
        x_norm = x_pixel / map_size
        y_norm = 1 - (y_pixel / map_size)
        
        # Convert normalized coordinates to longitude
        lon = x_norm * 360 - 180
        
        # Convert normalized coordinates to latitude
        lat = math.degrees(math.atan(math.sinh(math.pi * (2 * y_norm - 1))))
        
        return lat, lon
    
    def _getPolygon(self):
        """
        Returns
        -------
        polygon : string
            returns the corners of a square "polygon" as a string. This is the format used by MET url for ualf collection.

        """
        top, left = self.pixel_to_lat_lon(self.roi.left, self.roi.top, self.zoom_ov)
        bottom, right = self.pixel_to_lat_lon(self.roi.right, self.roi.bottom, self.zoom_ov)
        left_top = str(round(left,2))+" "+str(round(top,2))
        right_top = str(round(right,2))+" "+str(round(top,2))
        right_bottom = str(round(right,2))+" "+str(round(bottom,2))
        left_bottom = str(round(left,2))+" "+str(round(bottom,2))
        polygon = left_top+","+right_top+","+right_bottom+","+left_bottom
        return polygon

    def selectROI(self):
        """
        This function plots the overview map onto which a rectangular ROI is placed which can be used to determine what region ualf data is to be collected from.

        When closing the plot, a new rgb colored map is opened displaying the downloaded higher reaolution image covering an area slightly larger than the roi.

        """
        # We collect the upper left corner coordinates in tiles pixel values
        x_tile, y_tile = self.gmd_ov.getXY() 
        # Then we collect the hyperspyimage created when initiating gmd object and plot it
        self.map_ov_hs.plot() 
        # Next we create a hyperspy roi initiated at tile coordinates*256 to get to pixel value in pixel map (https://developers.google.com/maps/documentation/javascript/coordinates)
        if (self.roi == None):
            self.roi = hs.roi.RectangularROI(left=x_tile*256+100, top=y_tile*256+100, right=x_tile*256+500, bottom=y_tile*256+500)
        # And finally we plot an interactive roi on the hs figure for determining our region of interest:
        self.im_roi = self.roi.interactive(self.map_ov_hs, color="red")
        # We connect the figure to 
        plt.connect('close_event', self.onCloseROI)
        
    def onCloseROI(self, event):
        # Use the current parameters of the interactive ROI and find tile coordinates for zoomed in image corner:
        lat, lon = self.pixel_to_lat_lon(self.roi.left, self.roi.top, zoom=self.zoom_ov)
        # Default zoom to 12 for zoomed view for now. Should be detrmined by roi size 
        self.zoom_roi=12
        self.gmd_roi = GoogleMapDownloader(lat, lon, self.zoom_roi)
        # We find the pixel range of the roi for the new zoom level:
        px_range = abs(self.roi.right-self.roi.left)*2**(self.zoom_roi-self.zoom_ov)
        py_range = abs(self.roi.top-self.roi.bottom)*2**(self.zoom_roi-self.zoom_ov)
        # And then we determine how many tiles we need in new image to cover roi (Round up and add one tile to cover more than roi rather than less):
        xTiles = math.ceil(px_range/256+1)
        yTiles = math.ceil(py_range/256+1)
        self.roi_xTiles = xTiles
        self.roi_yTiles = yTiles
        
        roiX, roiY = self.gmd_roi.getXY() # This returns the lat lon pixels of upper left of cropped image
        self.roi_start_lat, self.roi_start_lon = self.pixel_to_lat_lon(256*roiX, 256*roiY, zoom=self.zoom_roi)
        self.roi_end_lat, self.roi_end_lon = self.pixel_to_lat_lon(256*(roiX+xTiles), 256*(roiY+yTiles), zoom=self.zoom_roi)
        print("Upper left corner: ("+str(self.roi_start_lat)+", "+str(self.roi_start_lon)+")")
        print("Upper left corner: ("+str(self.roi_end_lat)+", "+str(self.roi_end_lon)+")")
        self.gmd_roi.generateImage(tile_width=xTiles, tile_height=yTiles)
        self.map_roi_rgb = self.gmd_roi.getImageRGB()
        
        self.polygon = self._getPolygon()
        plt.figure()
        plt.imshow(self.map_roi_rgb)
        plt.show()
    
    def _httpGET_ualf_data_month(self, year, month):
        """
        A function collecting lightning data for a given month in ualf format.
        (https://opendata.smhi.se/apidocs/lightning/parameters.html)
    
        Parameters
        ----------
        year : int
            The year we want data from.
        month : int
            The month we want data from.
    
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
        # Client id below has to be collected by creating a user at MET
        return requests.get(endpoint, parameters, auth=(self.client_id,'')).text # returns data as text string
    
    def httpGET_ualf_data(self, ground_strikes, year_start, year_end, month_start=1, month_end=12):
        """

        Parameters
        ----------
        ground_strikes : bool
            True if only ground strikes are to be included in dataset, False if also cloud-cloud strikes are of interest.
        year_start : int
            First year to collect data from.
        year_end : int
            Last year to include data from.
        month_start : int, optional
            Month to start collection for each year at. Thought it would be nice to have the option to change it. The default is 1.
        month_end : int, optional
            Month to end collection for each year at. The default is 12.

        Returns
        -------
        None.

        """
        # header contains column parameters as defined by ualf format:
        header='d year month day hour min sec nanos lat lon pI multi nsens dof angle major minor chi2 rt ptz mrr cloud aI sI tI\n'
        year = year_start
        for year in range(year_start, year_end+1): # +1 as range does not include last int
            print(" ... Collecting data for ", year, "...")
            for month in tqdm(range(month_start, month_end+1)):
                header += self._httpGET_ualf_data_month(year, month)
        self.ualf = header
        self.ualf_crop = self.crop_ualf_data(header, ground_strikes)
    
    def ualf_to_array(self, ualf_string):
        ualf_rows = ualf_string.split("\n")
        i = 0
        for row in tqdm(ualf_rows):
            ualf_rows[i] = row.split(" ")
            i+=1
        return np.asarray(ualf_rows[:-1])
    
    def array_to_ualf(self, ualf_array):
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
    
    def save_ualf_data(self, filename):
        txt_file = self.array_to_ualf(self.ualf_crop)
        with open(filename, "w") as txt_save:
            txt_save.write(txt_file)
    
    def crop_ualf_data(self, ualf_data, only_ground_strikes=True):
        try:
            ualf_data.shape
            arr = ualf_data
        except:
            print("Converting string to array... This might take a while.")
            arr = self.ualf_to_array(ualf_data)
        
        polygon_numpy = np.asarray(re.split(r"[,\s]+", self.polygon), dtype=float)
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
    
    def getUalf(self):
        return self.ualf
    
    def getUalfCrop(self):
        return self.ualf_crop

    
    def plot_lightning_history(self):
        arr = self.ualf_crop
        start_year = int(arr[1:][0,1])
        end_year = int(arr[1:][-1,1])
        start_month = 1
        end_month = 12
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
            lightnings[date_i] += 1 # Counting the lightning strike for given month
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
        for i in range(len(non_zero_min_pI_indices)):
            ax2.text(non_zero_min_pI_indices[i], non_zero_min_pI_values[i]+offset, float(non_zero_min_pI_values[i]), c=min_pI_c)
        ax2.set_ylabel(f'Peak current [kA] (positive = {max_pI_c}, negative = {min_pI_c})', color='black')
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
        
    def plot_lightning_data(self):
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
        polygon = self.polygon
        lightning_array = self.ualf_crop
    
        poly_request = polygon.split(",")
        c1 = poly_request[0].split(" ")
        c2 = poly_request[1].split(" ")
        c3 = poly_request[2].split(" ")
        c4 = poly_request[3].split(" ")
        
        # Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        left = self.roi_start_lon 
        right = self.roi_end_lon 
        top = self.roi_end_lat 
        bottom = self.roi_start_lat

        extent = [left, right, top, bottom]  # [left, right, bottom, top]
        plt.imshow(self.map_roi_rgb, extent=extent)
        
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
        plt.scatter(float(c1[0]), float(c1[1]), label='c1', c='r')
        plt.scatter(float(c2[0]), float(c2[1]), label='c2', c='r')
        plt.scatter(float(c3[0]), float(c3[1]), label='c3', c='r')
        plt.scatter(float(c4[0]), float(c4[1]), label='c4', c='r')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # Add legend for years
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=year_to_color[year], markersize=10, label=str(year)) for year in years]
        plt.legend(handles=handles, title="Year")
        width, height = self.map_roi_rgb.size
        ax.set_aspect(2)
        
        N = lightning_array[1:].shape[0]/self.square_area()/(np.abs(years[-1]-years[0])+1)
        
        plt.title(f'{lightning_array.shape[0]} lightnings occured from {lightning_array[1,1]} to {lightning_array[-1,1]} within region. N = {N:.3f}')
        plt.show()
    
    def square_area(self):
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
        polygon_numpy = np.asarray(re.split(r"[,\s]+", self.polygon), dtype=float)
        lons = polygon_numpy[[0,2,4,6]]
        lats = polygon_numpy[[1,3,5,7]]
    
        # Create a spherical polygon
        polygon2 = sp.SphericalPolygon.from_lonlat([lons[0], lons[1], lons[2], lons[3]], [lats[0], lats[1], lats[2], lats[3]])
        return polygon2.area()*6378000**2*10**-6 # area formula returns steradians so I multiply with earth radius (A=4piR^2)
    
    def get_N(self):
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
        array = self.ualf_crop
        yrs = np.unique(array[1:,1].astype('int'))
        yr_span = np.abs(yrs[-1]-yrs[0])+1
        area = self.square_area()
        return array.shape[0]/area/yr_span

    def createAnimatedLightningData(self, interval, gif_name="ualf_gif"):
        arr = self.ualf_crop

        years = np.unique(arr[1:,1])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ims=[]
        
        # split = self.polygon.split(",")
        left = self.roi_start_lon #float(split[0].split(" ")[0])
        right = self.roi_end_lon #float(split[1].split(" ")[0])
        top = self.roi_end_lat #float(split[0].split(" ")[1])
        bottom = self.roi_start_lat #float(split[2].split(" ")[1])

        extent = [left, right, top, bottom]  # [left, right, bottom, top]
        plt.imshow(self.map_roi_rgb, extent=extent)

        # Define axis limits:
        polygon_numpy = np.asarray(re.split(r"[,\s]+", mug._getPolygon()), dtype=float)
        lons = polygon_numpy[[0,2,4,6]]
        lats = polygon_numpy[[1,3,5,7]]
        max_lat = max(lats)
        min_lat = min(lats)
        max_lon = max(lons)
        min_lon = min(lons)
        offset = 0.02
        ax.set_xlim(min_lon-offset, max_lon+offset)
        ax.set_ylim(min_lat-offset, max_lat+offset)
        ax.set_aspect(2) # This is just a rough value due to latitude, should be refined

        # We then need unique color for each year:
        num_colors = years[-1].astype('int')-years[0].astype('int')+1
        # Create unique colors from viridis color scheme:
        colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
        # And create a dictionary to collect color by year:
        year_to_color = {year: colors[i] for i, year in enumerate(years)}

        for iternum in range(len(years)):
            ttl = plt.text(0.5, 1.01, years[iternum], fontsize=15, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, color=year_to_color[years[iternum]])
            frame_arr = arr[np.where(arr[1:,1].astype('int') == years[iternum].astype(int))]#get_year_array(arr, years[iternum].astype(int))[1:] # removing header line
            x = frame_arr[1:,9].astype('float')
            y = frame_arr[1:,8].astype('float')
            ims.append([plt.scatter(x,y,marker='o',color=year_to_color[years[iternum]]), ttl])
            
        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False)
        ani.save(gif_name)
        plt.show()

mug = METuartDownloader()

mug.selectROI()

# This function wil download, sort out only data from ROI chosen above and convert data to array. It might take a while to finish:
ualf = mug.httpGET_ualf_data(False, 2018, 2024)
# mug.save_ualf_data("test.txt")

mug.createAnimatedLightningData(interval=1000, gif_name="Ryfylke.gif") # Interval = time (in ms) to wait before scene change

mug.plot_lightning_data()

mug.plot_lightning_history()

mug.square_area()

mug.get_N()


# Animation of lightnings per year: #################################################################

arr = mug.getUalfCrop()

years = np.unique(arr[1:,1])

fig = plt.figure()
ax = fig.add_subplot(111)
ims=[]

# split = self.polygon.split(",")
left = mug.roi_start_lon #float(split[0].split(" ")[0])
right = mug.roi_end_lon #float(split[1].split(" ")[0])
top = mug.roi_end_lat #float(split[0].split(" ")[1])
bottom = mug.roi_start_lat #float(split[2].split(" ")[1])

extent = [left, right, top, bottom]  # [left, right, bottom, top]
plt.imshow(mug.map_roi_rgb, extent=extent)

# Define axis limits:
polygon_numpy = np.asarray(re.split(r"[,\s]+", mug._getPolygon()), dtype=float)
lons = polygon_numpy[[0,2,4,6]]
lats = polygon_numpy[[1,3,5,7]]
max_lat = max(lats)
min_lat = min(lats)
max_lon = max(lons)
min_lon = min(lons)
offset = 0.001
ax.set_xlim(min_lon-offset, max_lon+offset)
ax.set_ylim(min_lat-offset, max_lat+offset)
ax.set_aspect(2)

# We then need unique color for each year:
num_colors = years[-1].astype('int')-years[0].astype('int')+1
# Create unique colors from viridis color scheme:
colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
# And create a dictionary to collect color by year:
year_to_color = {year: colors[i] for i, year in enumerate(years)}

for iternum in range(len(years)):
    ttl = plt.text(0.5, 1.01, years[iternum], fontsize=15, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes,color=year_to_color[years[iternum]])
    frame_arr = arr[np.where(arr[1:,1].astype('int') == years[iternum].astype(int))]#get_year_array(arr, years[iternum].astype(int))[1:] # removing header line
    x = frame_arr[1:,9].astype('float')
    y = frame_arr[1:,8].astype('float')
    ims.append([plt.scatter(x,y,marker='o',color=year_to_color[years[iternum]]), ttl])
    
ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.show()
