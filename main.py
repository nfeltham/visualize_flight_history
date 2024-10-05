import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
from cartopy.io.img_tiles import OSM
import numpy as np
import os
import geopy.distance as gd
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from datetime import datetime


class plotFlightPath():
    imagery = OSM()
    projection = ccrs.PlateCarree()
    
    # max and min zoom level
    size_max = 12.5
    size_min = 2

    # scale the number of points plotted in each flight
    num_points_scale = 5

    # scale the spacing of the points : higher number means less points in the middle of the flight
    line_sample_power = 2

    # what percentage of the flight the zoom occurs 0-50
    size_taper = 40 

    # frame rate of the resulting video 
    frame_rate = 25

    # name of the resulting video
    video_name = "output"

    # keep or delete the images created
    keep_images = True

    # where the images will be stored
    output_dir = "./.tmp_images"

    # airline : color for visualizing the airplane marker
    aircraft_colors = {"UA":"MediumSlateBlue", "AA":"IndianRed", "LH": "Yellow", "DL": "DarkRed", "KLM": "Pink", "EI":"Green"}
     # aircraft_colors = {"UA":"b", "AA":"r", "LH": "y", "DL": "g", "KLM": "c"}

    # lat, lon of airports
    airport_coords = {"CLT" : [35.2145, -80.9488],
                      "SDF" : [38.1707, -85.7308],
                      "FRA" : [50.0380, 8.5622],
                      "CMH" : [39.9999, -82.8872],
                      "EWR" : [40.6895, -74.1745],
                      "ORD" : [41.9802, -87.9090],
                      "ATW" : [44.2605, -88.5111],
                      "CVG" : [39.0514, -84.6671],
                      "DFW" : [32.7079, -96.9209],
                      "SFO" : [37.6192, -122.3816],
                      "FMO" : [52.1343, 7.68343],
                      "IAD" : [38.9523, -77.4586],
                      "IAH" : [29.9931, -95.3416],
                      "CHS" : [32.8917, -80.0395],
                      "AMS" : [52.3130, 4.7725],
                      "ATL" : [33.6361, -84.4294],
                      "FAT" : [36.7782, -119.7165],
                      "LAS" : [36.0831, -115.1482],
                      "MUC" : [48.3536, 11.7832],
                      "PIT" : [40.4929, -80.2373],
                      "BRE" : [53.0480, 8.7859],
                      "GUC" : [38.3202, -106.5559],
                      "DEN" : [39.8563, -104.6764],
                      "DUB" : [53.4256, -6.2574],
                      "BWI" : [39.1776, -76.6684],
                      "HKG" : [22.3135, 113.9137],
                      }

    def __init__(self):
        self._get_plane_marker()
        self._create_legend()

        
    def _get_num_points(self, start, end, airline):
        """
        start: (lat, lon) of start point
        end: (lat, lon) of end point
        scale: multiple scale value for number of points.

        The number of points is scaled by the square root of the distance 
        """
        
        dist = gd.geodesic(start, end).miles
        
        max_map_size = 6 + dist/200
        self.size_max = min(max_map_size, 12.5)

        num_points = int(self.num_points_scale * np.sqrt(dist) / 10)
        if airline is None:
            num_points = int(num_points/2)
        
        return num_points

    
    def _get_plane_marker(self):
        """
        initialize the airplane marker from svg file
        """
        
        plane_path, attributes = svg2paths('plane-icon.svg')
        self.plane_marker = parse_path(attributes[0]['d'])
        self.plane_marker.vertices -= self.plane_marker.vertices.mean(axis=0)
        self.plane_marker = self.plane_marker.transformed(mpl.transforms.Affine2D().rotate_deg(45))


    def _create_legend(self):
        """
        create the legened for the plots. Currently just a bar becuase I cant figure out how to make it the airplane marker 
        """
        
        full_names = {"UA":"United", "AA":"American", "LH": "Lufthansa", "DL": "Delta", "KLM": "KLM", "EI":"Aer Lingus"}
        self.legend_elements = [mpl.patches.Circle((0,0), radius=.01, color=self.aircraft_colors[key], label=full_names[key]) for key in self.aircraft_colors.keys()]

        
    def _convert_date(self, date):
        """
        convert the date from m/d/y to something nicer for visualizing
        """
        
        datetime_object = datetime.strptime(date, '%m/%d/%Y')
        return datetime_object.strftime('%d %b %Y')
        
        
    
    def _nonlinspace(self, xmin, xmax, n=10, power=2):
        """
        Intervall from xmin to xmax with n points, the higher the power, the more dense towards the ends

        Not used, I think sigmoidspace works better
        """
        
        xm = (xmax - xmin) / 2
        x = np.linspace(-xm**power, xm**power, n)
        array = np.sign(x)*abs(x)**(1/power) + xm + xmin
        if xmin > xmax:
            array = np.flip(array)
        return array


    def _sigmoidspace(self, low, high, n):
        """
        Space the points out so more are clustered around the airports, and less in the middle of the flight. Makes it appear like the plane is speeding up and makes visualizing better
        """
        
        raw = np.tanh(np.linspace(-self.line_sample_power, self.line_sample_power, n))
        return (raw-raw[0])/(raw[-1]-raw[0])*(high-low)+low

    
    def _calc_map_size(self, index, num_taper_points, num_points):
        """
        Calculate how much to zoom in based on how close you are to the airport and what your taper size is
        """
        
        if index < num_taper_points:
            size=np.linspace(self.size_min, self.size_max, num_taper_points)[index]
        elif index >= num_points - num_taper_points:
            size=np.linspace(self.size_max, self.size_min, num_taper_points)[index - (num_points-num_taper_points)]
        else:
            size = self.size_max
        return size

    
    def _calc_imagery_level(self, map_size):
        """
        Calculate the image detail. More zoomed in=more detail
        """
        
        if map_size <= 4:
            return 7
        elif map_size <= 7:
            return 6
        else:
            return 5

        
    def _calc_marker_angle(self, start, end):
        """
        Calculate the heading of the airplane for plotting
        """
        
        return np.rad2deg(np.arctan2(end[0]-start[0], end[1]-start[1]))

        
    def _make_video(self):
        """
        Create the video from the images created
        """
        
        os.system(f"ffmpeg -framerate {self.frame_rate} -i {self.output_dir}/img_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p {self.video_name}.mp4")
        if not self.keep_images:
            os.system(f"rm -r {output_dir}")
        

    def _plot_map(self, lon_marker, lat_marker, map_size, index, plot_index, title, airline):
        """
        Main plotting function. Plot each individual point. 
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection=self.imagery.crs)

        if airline is not None:
            ax.plot(lon_marker[index], lat_marker[index], self.aircraft_colors[airline], transform=self.projection, marker=self.plane_marker, markersize=10)

        ax.set_extent([lon_marker[index]-map_size*1.25, lon_marker[index]+map_size*1.25, lat_marker[index]+map_size, lat_marker[index]-map_size], self.projection)
        ax.add_image(self.imagery, self._calc_imagery_level(map_size))

        fig.legend(handles=self.legend_elements, loc='center right')

        plt.title(self._convert_date(title) if title != "-" else title)
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/img_{plot_index}.png", dpi=250)
        plt.close()


    def _init_check_data(self,values):
        """
        Check to make sure all airlines are available before starting.
        """
        if not values[3] in self.aircraft_colors.keys() and not values[3] in self.missing_airlines:
           self.missing_airlines.append(values[3])

            
    def _check_non_consecutive_city(self, list_of_coords):
        """
        Add points between cities when the takeoff city is not the same as the previous landing city. Makes the video less jumpy and easier to follow. 
        """
        
        new_list = []
        for i in range(len(list_of_coords)-1):
            new_list.append(list_of_coords[i])
            if list_of_coords[i][1] != list_of_coords[i+1][0]:
                new_list.append([list_of_coords[i][1], list_of_coords[i+1][0], "-", None])
        new_list.append(list_of_coords[-1])
        
        return new_list

    
    def single_trip_plot(self, start_location, end_location, title, airline, start_index=0, single_trip=True):
        """
        Calculate all the parameters and call the plotting function for one flight 
        """
        
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        marker_angle = self._calc_marker_angle(start_location, end_location)
        self.plane_marker = self.plane_marker.transformed(mpl.transforms.Affine2D().rotate_deg(marker_angle))

        num_points = self._get_num_points(start_location, end_location, airline)
        lat_marker = self._sigmoidspace(start_location[0], end_location[0], n=num_points)
        lon_marker = self._sigmoidspace(start_location[1], end_location[1], n=num_points)
        
        num_taper_points = int(num_points * self.size_taper / 100)
 
        for i in range(num_points):
            map_size = self._calc_map_size(i, num_taper_points, num_points)
            self._plot_map(lon_marker, lat_marker, map_size, i,  i+start_index, title, airline)
            
        self.plane_marker = self.plane_marker.transformed(mpl.transforms.Affine2D().rotate_deg(-marker_angle))

        if single_trip:
            self._make_video()
        
        return num_points

    
    def plot_multiple_trips(self, list_of_coords):
        """
        Plot multiple flights given a list of flight data
        """
        plot_index = 0

        for i in range(len(list_of_coords)):
            plot_index += self.single_trip_plot(list_of_coords[i][0], list_of_coords[i][1], list_of_coords[i][2], list_of_coords[i][3], start_index=plot_index, single_trip=False)

        self._make_video()


    def plot_from_csv(self, csvFile):
        """
        Plot all flights listed in a given csvFile
        """
        
        csvFile = open(csvFile, "r")

        header = csvFile.readline()
        list_of_coords, self.missing_airlines = [], []

        while True:
            line = csvFile.readline()
            if line == "":
                break
            values = [text.strip() for text in line.split(",")]
            self._init_check_data(values)
            newRow = [self.airport_coords[values[1]], self.airport_coords[values[2]], values[0], values[3]]
            list_of_coords.append(newRow)

        if len(self.missing_airlines) != 0:
            print(f"Missing airline color(s): {self.missing_airlines}")
            exit()
            
        list_of_coords = self._check_non_consecutive_city(list_of_coords)
        
        self.plot_multiple_trips(list_of_coords)
            
            
            
if __name__ == "__main__":
    csv_name = "flight_data.csv"
    
    ad = plotFlightPath()
    # ad.projection = ccrs.Geodetic()

    ad.single_trip_plot(ad.airport_coords["SFO"], ad.airport_coords["HKG"], "08/23/2024", "UA", start_index=0, single_trip=True)
    
    #ad.video_name = "output2"
    #ad.plot_from_csv(csv_name)
