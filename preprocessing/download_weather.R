
library(data.table)
library(raster)
library(ggplot2)
library(ncdf4)

#======================================================================#
## define functions
#======================================================================#
`%ni%` <- Negate(`%in%`) 

### functions to get meteorology data

# download the necessary met files, 20th century reanalysis
downloader.fn <- function(filename){
  
  fileloc <- file.path('.', 'data', 'weather', 'NARRData')
  
  # create directory to store in
  dir.create(fileloc, 
             recursive = T, 
             showWarnings = F)
  
  # name variable, filenames
  varname_NOAA <- gsub("\\..*", "", filename)
  file_NOAA <- file.path(fileloc, filename)
  url_NOAA <- paste0("ftp://ftp.cdc.noaa.gov/Datasets/NARR/Monthlies/monolevel/", filename)
  
  if (!file.exists(file_NOAA)){
    download.file(url = url_NOAA, destfile = file_NOAA)
  }
  
  hpbl_rasterin <- brick(x = file_NOAA, varname = varname_NOAA)
  
  return(hpbl_rasterin)
}


#' Read netcdf file and extract date
#'
#' \code{subset_nc_date}  takes as input a netcdf file location, date, and variable name
#' and outputs the data as a raster layer.
#'
#' @param hpbl_file netcdf file path
#' @param varname variables name
#' @param vardate variable date
#' @return raster layer of input file, variable, and date

subset_nc_date <- function( hpbl_file = NULL,
                            hpbl_brick = NULL,
                            varname = NULL,
                            vardate){
  
  if( (is.null( hpbl_file)  & is.null( hpbl_brick)) |
      (!is.null( hpbl_file) & !is.null( hpbl_brick)))
    stop( "Uh oh! Please define EITHER hpbl_file OR hpbl_brick")
  
  Sys.setenv(TZ='UTC')
  
  if( !is.null( hpbl_file))
    rasterin <- rotate( brick( hpbl_file, varname = varname ))
  if( !is.null( hpbl_brick))
    rasterin <- hpbl_brick
  
  #get time vector to select layers
  dates <- names( rasterin)
  dates <- gsub( 'X', '', dates)
  dates <- gsub( '\\.', '-', dates)
  
  # Get first day of the month for vardate
  vardate_month <- as.Date( paste( year( vardate),
                                   month( vardate),
                                   '01',
                                   sep = '-'))
  
  #select layer
  layer <- which( dates == vardate_month)
  if( length( layer) == 0)
    stop( "Cannot match the dates of PBL raster file. Did you set the time zone to UTC before reading it in? (Sys.setenv(TZ='UTC'))")
  
  rastersub <- raster::subset(rasterin, subset = layer)
  
  return(rastersub)
}

# extract the year of interest
extract_year.fn <- function(raster.in = list.met[[1]], year.in = 2005) {
  
  # name months 1:12 for extracting from raster
  names.months <- paste0(year.in, '-',
                         formatC(1:12, width = 2, flag = '0'), '-', '01')
  
  # extract monthly dates using function from hyspdisp
  raster.sub <- subset_nc_date(hpbl_brick = raster.in,
                               vardate = names.months)
  
  # take annual mean
  raster.sub.mean <- stackApply(raster.sub, indices = rep( 1, 12), fun = mean)
  

  return(raster.sub.mean)
}


extract_month.fn <- function(
  raster.in = list.met[[1]],
  year.in = 2005,
  month.in = 1
){
  # name months 1:12 for extracting from raster
  names.months <- paste0(year.in, '-',
                         formatC(1:12, width = 2, flag = '0'), '-', '01')
  
  # extract monthly dates using function from hyspdisp
  raster.sub <- subset_nc_date(hpbl_brick = raster.in,
                               vardate = names.months[month.in])
  
  return(raster.sub)
}


# trim data over US, combine into data.table, create spatial object
usa.functioner <- function(year.in = 2005,
                           list.met,
                           month.in = NULL){
  
  # extract year
  if (is.null( month.in)) {
    mets <- brick(lapply(list.met,
                       extract_year.fn,
                       year.in = year.in))
  } else {
    mets <- brick(lapply(list.met,
                       extract_month.fn,
                       year.in = year.in,
                       month.in = month.in))
    
  }

  
  crs.usa <- crs("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +a=6370000 +b=6370000")

  mets <- projectRaster(mets, crs = crs.usa)
  return(mets)
}

#======================================================================#
## download monthly met data
#======================================================================#

# see all variables at
# https://psl.noaa.gov/data/gridded/data.narr.monolevel.html

# define the layer names, do the actual downloading
Sys.setenv(TZ='UTC')
layer.names <- c("air.2m.mon.mean.nc", # temperature at 2m
                 "apcp.mon.mean.nc", # total precip,
                 "rhum.2m.mon.mean.nc", # relative humidity
                 "vwnd.10m.mon.mean.nc", # v (north-south) wind component
                 "uwnd.10m.mon.mean.nc") # u (east-west) wind component
                  
names(layer.names) <- c("temp", "apcp", "rhum", "vwnd", "uwnd")

# do the data downloading
# choose NARR for best spatial resolution  over the US
# monthly data
list.met <- lapply(layer.names, downloader.fn)

# get all data and store in binary format
years = 2000:2015
months = 1:12
controls = list()

for (y in years) {
  for (m in months) {
    ym = sprintf("%s%02d", y, m)
    controls[[ym]] = usa.functioner(y, list.met, m)
    print(paste("finished", ym))
  }
}

saveRDS(controls, "data/weather/weather.rds")
readr::write_csv(as.data.frame(layer.names), "data/weather/weather_names.csv")
