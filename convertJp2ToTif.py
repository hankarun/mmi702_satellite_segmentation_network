import glob
import gdal
import os

def convertToTif(filename):
    filen = os.path.basename(f)
    targetFilename = filename + ".tif"
    targetWarped = filename + ".wrapped.tif"
    srcDs = gdal.Open(filename)
    print("jp2 to geotiff")
    gdal.Translate(targetFilename, srcDs)
    srcDs = gdal.Open(targetFilename)
    print("warping to EOSG:4326")
    gdal.Warp(targetWarped, srcDs, dstSRS="EPSG:4326")
    srcDs = gdal.Open(targetWarped)
    os.remove(targetFilename)
    print("clipping")
    gdal.Translate(targetFolder + filen + ".tif", srcDs, projWin=clippingArea)
    os.remove(targetWarped)


clippingArea = [32.818549029, 39.889423097, 32.85881772, 39.865975505]
currentFolder = "/mnt/c/Users/aliis/Desktop/702_Project/sentinelRaw/bands/data3"
targetFolder = "/mnt/c/Users/aliis/Desktop/702_Project/sentinelRaw/clipped/data3/"
jp2Files = glob.glob(currentFolder + "/**/*.jp2", recursive=True)

for f in jp2Files:    
    convertToTif(f)