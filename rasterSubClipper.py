import glob
import gdal
import os
from gdalconst import GA_ReadOnly

stepX = 6
stepY = 4

currentFolder = "/mnt/c/Users/aliis/Desktop/702_Project/sentinelRaw/clipped/data3"
targetFolder = "/mnt/c/Users/aliis/Desktop/702_Project/sentinelRaw/input/data3/"
tifFiles = glob.glob(currentFolder + "/**/*.tif", recursive=True)

for f in tifFiles:
    targetFile = gdal.Open(f, GA_ReadOnly)
    geoTransform = targetFile.GetGeoTransform()
    print(geoTransform)
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * targetFile.RasterXSize
    miny = maxy + geoTransform[5] * targetFile.RasterYSize
    print(minx, miny, maxx, maxy)
    xStepSize = geoTransform[1] * targetFile.RasterXSize / stepX
    yStepSize = geoTransform[5] * targetFile.RasterYSize / stepY
    for x in range(stepX):
        for y in range(stepY):
            filen = os.path.basename(f)
            filePiece = "{}{}_x_{}_y_{}.tif".format(targetFolder,filen,x,y)
            print(filePiece)
            proj = [minx + x * xStepSize, miny - y * yStepSize, minx + (x + 1) * xStepSize, miny - (y + 1) * yStepSize]
            print(proj)
            gdal.Warp(filePiece, targetFile, outputBounds=proj)