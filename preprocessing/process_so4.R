library(parallel)
library(tidyverse)
library(raster)
library(rgdal)
library(stringr)

# Cluster for parallel computing
num_processes = 3


nrow = 128
ncol = 256
datadir = "./data/SO4"

xmin = -135.0
xmax = -60.0
ymin = 20.0
ymax = 52.0


ext = extent(xmin, xmax, ymin, ymax)

# Define the projection to be used
crs_wgs84=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

# 1 convert asc writer to .tif
# this should be done in the download data module
# Find asc rasters  

datadir = "./data/SO4"
paths = list.files(
  datadir,
  pattern = "asc.zip$",
  recursive = TRUE,
  include.dirs = FALSE
)

# if (length(grids) > 0) {
#   paths = paste0(datadir, grids)
# } else {
#   paths = character(0)
# }

crswgs84 = sp::CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

# Convert all rasters to .tiff if necessary
print("Converting .asc to .tif...")

dir.create(paste(datadir, "tif", sep = "/"))

cl = makeCluster(num_processes)
# pblapply(
# lapply(
parLapply(
  cl,
  paths,
  function(p) {
    p = stringr::str_replace(p, ".zip", "")
    path_tif = paste0(stringr::str_sub(p, end=-5), ".tif")
    datadir = "./data/SO4"
    crswgs84 = sp::CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    nrow = 128
    ncol = 256
    xmin = -135.0
    xmax = -60.0
    ymin = 20.0
    ymax = 52.0
    ext = raster::extent(xmin, xmax, ymin, ymax)
    rast_canvas = raster::raster(nrow=nrow, ncol= ncol, ext=ext, crs=crswgs84)

    if (!file.exists(paste(datadir, "tif", path_tif, sep="/"))) {
      print(paste("Processing", p))
      print(paste("    unzip target", p))
      unzip(paste(datadir, paste0(p, ".zip"), sep="/"), files=p, exdir=paste(datadir, "tif", sep="/"), overwrite = TRUE)
      print(paste("    reading", "..."))
      rast = raster::raster(paste(datadir, "tif", p, sep="/"))
      raster::projection(rast) = crswgs84
      # print(paste("    writing to", paste(datadir, "tif", path_tif, sep="/"), "..."))
      # raster::writeRaster(rast, paste(datadir, "tif", path_tif, sep="/"), overwrite=TRUE)
      #
      print(paste("    resizing..."))
      rast = raster::aggregate(rast, fun=mean, fact=16, na.rm=TRUE)
      # print(paste("    projecting..."))
      # rast = raster::projectRaster(rast, crs=crswgs84, method="ngb")
      # rast = raster::crop(rast, ext)
      print(paste("    projecting in 256x128 grid..."))
      rast = raster::resample(rast, rast_canvas)
      ym = stringr::str_sub(p, -24, -19)
      tgt_file = sprintf("%s.tif", ym)
      print(paste("    writing to", paste(datadir, "tif", tgt_file, sep="/")))
      raster::writeRaster(rast, paste(datadir, "tif", tgt_file, sep="/"), overwrite=TRUE)
      rm(rast)
      file.remove(paste(datadir, "tif", p, sep="/"))
    }
  # }, cl=cl
  }
)
stopCluster(cl)
