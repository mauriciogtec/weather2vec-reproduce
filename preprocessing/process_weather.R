library(parallel)
library(tidyverse)
library(raster)
library(rgdal)
library(stringr)

nrow = 128
ncol = 256
remove_wspeed_and_phi = TRUE

# obtained from the download_met_data.R function
controls = readRDS("data/weather.rds")
dir.create("data/weather", showWarnings = FALSE)

# crs_usa = raster::crs("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +a=6370000 +b=6370000")
crs_wgs84 = raster::crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")


xmin = -135.0
xmax = -60.0
ymin = 20.0
ymax = 52.0

ext = extent(xmin, xmax, ymin, ymax)

rast_canvas = raster::raster(
  nrow=nrow, ncol=ncol, ext=ext, crs=crs_wgs84
)


for (t in 1:length(controls)) {
  nm = names(controls)[t]
  if (!(substr(nm, 1, 4) %in% as.character(2000:2015))) {
    print(paste("Skipping", nm))
    next
  }
  print(paste("Processing", nm))
  rast = controls[[t]]
  rast = raster::projectRaster(rast, crs=crs_wgs84)
  rast = raster::crop(rast, ext)
  # plot(rast)
  rast = raster::resample(rast, rast_canvas)
  # plot(rast)
  fname = sprintf("data/weather/%s.tif", nm)
  raster::writeRaster(rast, fname, overwrite=TRUE)
}