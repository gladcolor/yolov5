from pyproj import Transformer, Proj, transform
import matplotlib.pylab as plt
import math
import pandas as pd
import numpy as np
import os
import glob
import glob
import os
import math
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math
from math import *
from scipy.stats import norm
from sklearn.metrics import r2_score


import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter


ground_truth = r"..\\Data\2_Elevation_Certificates\Elevation_certificates.csv"
EC_df = pd.read_csv(ground_truth)
EC_df = EC_df.set_index("ID")


detected_dir = r'..\Data\3_Detected_doors'



dets_txts = glob.glob(os.path.join(detected_dir, "*.txt"))
print(len(dets_txts))



column_names = ["image", "panoId","EC_id", "confidence", "col","bottom_row","top_row",
                    "FFE_gsv_m","EC_FFE_m","error_m","delta_h","distance",
                "sink_distance","camera_h","camera_dem_m","phi", "pano_lat", "pano_lon",
               "target_lat", "target_lon", "target_elev", "image_date", 'edge']

        



def is_height_floor(height_list, thres_m=2):
    max_h = max(height_list)
    min_h = min(height_list)
    range_h = max_h - min_h
    result = [False] * len(height_list)
    print(max_h)
    if range_h > thres_m:
        print("Found doors on 2nd floor.")
        for idx, h in enumerate(height_list):
            if h - min_h > thres_m:
                result[idx] = True

    return result

def is_at_edge(x, y, w, h, edge_thres=0.01):
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    top = y - h / 2
    bottom = y + h / 2
    left = x - w / 2
    right = x + w / 2

    if edge_thres > left:
        return "left"
    if right > (1 - edge_thres):
        return "right"

    if edge_thres > top:
        return "top"
    if bottom > (1 - edge_thres):
        return "bottom"

    return "Not_at_edge"



def castesian_to_shperical(col, row, fov_h, height, width):  # yaw: set the heading, pitch
    """
    Convert the row, col to the  spherical coordinates 
    :param row, cols: 
    :param fov_h:
    :param height:
    :param width:
    :return:
    """
     
    col = col - width/2  # move the origin to center
    row = height/2 -row
 
    fov_v = atan((height * tan((fov_h / 2)) / width)) * 2
     
    
    r = (width/2)/tan(fov_h/2)
     
    s = sqrt(col**2 + r**2)
    
    theta = atan(row/s)
    phi = atan(col/r)
    

     
    return phi, theta


def target_to_lonlat(heading_deg, distance_m, car_lon, car_lat):
    
    heading_rad = math.radians(heading_deg)
    x = distance_m * math.sin(heading_rad)
    y = distance_m * math.cos(heading_rad)
    
    csr_string = f"+proj=tmerc +lat_0={car_lat} +lon_0={car_lon} +datum=WGS84 +units=m +no_defs"
    inProj =  Proj(csr_string)
    
    outProj = Proj('epsg:4326')  # WGS84

    
    target_lat, target_lon = transform(inProj, outProj, x, y)
    
#     transformer = Transformer(inProj, outProj)
#     transformer.transform(12, 12)

    
    return target_lon, target_lat

def process_detected():
    measurement_df = pd.DataFrame(columns=column_names)

    target_id = 0

    for idx, txt in enumerate(dets_txts[0:]):

        f = open(txt, 'r')
        lines = f.readlines()

        lines = [line.replace(" \n", "") for line in lines]

        objects_fileds = [line.split(' ') for line in lines]


        fov_h = math.pi/6



        img_name = txt.replace(".txt", '.jpg')
        img = Image.open(img_name)

        img_w, img_h = img.size

        img_name = os.path.basename(txt)[:-4]

    #     panoId = extract_panoId_frm_name(img_name)
    #     panoId = extract_panoId_frm_Latlon_name(img_name)
        basename = os.path.basename(txt)
        EC_id = basename.split("_")[0]
        EC_id = int(EC_id)
        EC_FFE = EC_df.loc[EC_id]['FFE_ft']


        img_heading_deg = basename.replace(".txt", '').split("_")[-1]

        panoId = EC_df.loc[EC_id]['panoId']


        image_date = EC_df.loc[EC_id]['image_date']

    #     if image_date < "2014-01-01":
    #         continues

        camera_h = EC_df.loc[EC_id]['camera_height_m']
        camera_dem_m = EC_df.loc[EC_id]['DEM_ft'] * 0.3048
        pano_lat = EC_df.loc[EC_id]['pano_lat']
        pano_lon = EC_df.loc[EC_id]['pano_lng']


        for o in objects_fileds:
            try:

                obj_id = int(o[0])
                cxywh = o[1:]
                conf, x, y, w, h = [float(n) for n in cxywh]

                if obj_id == target_id:

                    print(idx, "/", len(dets_txts))

                    print("EC_id, panoId", EC_id, panoId)

                    current_row = len(measurement_df)

                    top_row = img_h * (y - h/2)
                    bottom_row = img_h * (y + h/2)
                    col = img_w * x



                    top_agls = castesian_to_shperical(col, top_row, fov_h, img_h, img_w)
                    bottom_agls = castesian_to_shperical(col, bottom_row, fov_h, img_h, img_w)
                    phi = bottom_agls[0]



                    a1 = top_agls[1]
                    a2 = bottom_agls[1]



                    door_h = 2.03

                    distance = door_h * cos(a1) * cos(a2) / sin(a1 - a2)  # meters

                    sink_distance = distance * tan(a2)


                    delta_h = camera_h + sink_distance

                    FFE_gsv_m = delta_h + camera_dem_m

                    EC_FFE_m = EC_FFE * 0.3048

                    error_m = abs(FFE_gsv_m - EC_FFE_m)
                    #error_m =FFE_gsv_m - EC_FFE_m

                    target_heading_deg = float(img_heading_deg) + float(math.degrees(phi))

                    edge = is_at_edge(x, y, w, h)

                    target_lon, target_lat = \
                                          target_to_lonlat(target_heading_deg, distance, pano_lon, pano_lat)

                    spherical_resolution_deg = 180/4096
                    spherical_resolution_rad = math.radians(spherical_resolution_deg)
                    vertical_resolutoin = distance * (math.tan(top_agls[1] + spherical_resolution_rad) -
                                                      math.tan(top_agls[1]))

                    print("distance: ", distance)
                    print("error_m: ", error_m)

                    measurement_df.at[current_row, 'image'] = img_name + '.jpg'
                    measurement_df.at[current_row, 'panoId']   = panoId
                    measurement_df.at[current_row, 'EC_id']    = EC_id
                    measurement_df.at[current_row, 'confidence'] = conf
                    measurement_df.at[current_row, 'col']     = int(col)

                    measurement_df.at[current_row, 'bottom_row'] = int(bottom_row)
                    measurement_df.at[current_row, 'top_theta'] = top_agls[1]
                    measurement_df.at[current_row, 'bottom_theta'] = bottom_agls[1]

                    measurement_df.at[current_row, 'FFE_gsv_m'] = FFE_gsv_m
                    measurement_df.at[current_row, 'EC_FFE_m'] = EC_FFE_m
                    measurement_df.at[current_row, 'error_m']  = error_m
                    measurement_df.at[current_row, 'delta_h']  = delta_h
                    measurement_df.at[current_row, 'distance'] = distance
                    measurement_df.at[current_row, 'sink_distance'] = sink_distance
                    measurement_df.at[current_row, 'camera_h'] = camera_h
                    measurement_df.at[current_row, 'camera_dem_m'] = camera_dem_m
                    measurement_df.at[current_row, 'phi'] = phi
                    measurement_df.at[current_row, 'pano_lat'] = pano_lat
                    measurement_df.at[current_row, 'pano_lon'] = pano_lon
                    measurement_df.at[current_row, 'target_lat'] = target_lat
                    measurement_df.at[current_row, 'target_lon'] = target_lon
                    measurement_df.at[current_row, 'target_elev'] = FFE_gsv_m
                    measurement_df.at[current_row, 'image_date'] = image_date
                    measurement_df.at[current_row, 'edge'] = edge



            except Exception as e:
                print("Error: " , e)
                continue



    measurement_df["FFE_gsv_m"] = measurement_df["FFE_gsv_m"].astype(float)

    lowest_door_idx = measurement_df[measurement_df['edge'] == 'Not_at_edge'].groupby("EC_id")["FFE_gsv_m"].idxmin()

    measurement_df["is_lowest_door"] = 'N'
    measurement_df.loc[lowest_door_idx, ["is_lowest_door"]] = "Y"

    measurement_df = measurement_df[measurement_df["is_lowest_door"] == 'Y']

    measure_file = os.path.join(detected_dir, 'measurements.csv')
    measurement_df.to_csv(measure_file, index=False)


def draw_historgam():
    measure_file = os.path.join(detected_dir, 'measurements.csv')
    measurement_df = pd.read_csv(measure_file)
    errors = measurement_df[measurement_df["is_lowest_door"] == 'Y']['error_m'].tolist()
    errors = np.array(errors)
    mu, std = norm.fit(errors)
    weights = np.ones_like(errors) / float(len(errors))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    GSV_LFE = measurement_df["FFE_gsv_m"]
    ECCs = measurement_df["EC_FFE_m"]


    ax2.scatter(ECCs, GSV_LFE, s=7)
    ax2.axis('scaled')
    ax2.set(xlabel='LFE (m)', ylabel="GSV LFE (m)", title=r'(b) LFE - GSV_LFE Scatter')
    ax2.set_xlim([1, 8.8])
    ax2.set_ylim([1, 8.8])
    z = np.polyfit(ECCs, GSV_LFE, 1)
    p = np.poly1d(z)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(GSV_LFE, p(ECCs)):0.3f}$"
    print(text)

    ax2.plot(ECCs, p(ECCs), 'r--')
    ax2.text(0.05, 0.95, text, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top')


    bins = [0.1 * i for i in range(math.ceil(errors.max()/0.1) + 1)]
    print(bins)

    ax.hist(errors, bins=bins, alpha=0.7, rwidth=0.8,  weights=weights )
    xmin, xmax = plt.xlim()
    print(math.ceil(errors.max()/0.1))
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    title = "(a) Error historgram"

    yticks = mtick.PercentFormatter(xmax=1, decimals=0)
    ax.yaxis.set_major_formatter(yticks)
    ax.set(xlabel='Error (m)', ylabel="Frequency", title=title)
    mean = errors.mean()
    median = np.median(errors)
    std = errors.std()
    rmse = np.sqrt((errors**2).mean())
    text = f"$mean={mean:.3f}$\n$median={median:.3f}$\n$std={std:.3f}$\n$RMSE={rmse:.3f}$\n"

    print(text)

    ax.text(0.8, 0.35, text, fontsize=14, verticalalignment='top')


    plt.show()


def iou(bbox1, bbox2, vertical=True, horizontal=True):  # bbox: (top, right, bottom, left)
    top1, right1, bottom1, left1 = bbox1
    top2, right2, bottom2, left2 = bbox2




if __name__ == '__main__':
   process_detected()
   draw_historgam()