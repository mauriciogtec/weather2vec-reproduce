
library(rnaturalearth)
library(data.table)
library(raster)
# library(splitr)
# # library(hyspdisp)
library(ggplot2)
library(ncdf4)


#======================================================================#
## define functions
#======================================================================#
`%ni%` <- Negate(`%in%`) 


### functions to get meteorology data

# download the necessary met files, 20th century reanalysis
downloader.fn <- function(filename,
  dataset = c('20thC_ReanV2c', 'ncep.reanalysis.derived', 'NARR')){
  
  if (length(dataset) > 1)
    dataset <- dataset[1]
  
  fileloc <- file.path('.', 'data', 'WindData', dataset)
  
  # create directory to store in
  dir.create(fileloc, 
             recursive = T, 
             showWarnings = F)
  
  # name variable, filenames
  varname_NOAA <- gsub("\\..*", "", filename)
  file_NOAA <- file.path(fileloc, filename)
  
  # define URL
  if(dataset == '20thC_ReanV2c'){
    
    # Source: https://www.esrl.noaa.gov/psd/data/gridded/data.20thC_ReanV2c.monolevel.mm.html
    
    url_NOAA <- paste0("ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Monthlies/gaussian/monolevel/", filename)
  
  } else if (dataset == 'ncep.reanalysis.derived'){
    
    url_NOAA <- paste0("ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/surface/", filename)
  
  } else if (dataset == 'NARR'){
    
    url_NOAA <- paste0("ftp://ftp.cdc.noaa.gov/Datasets/NARR/Monthlies/monolevel/", filename)
  }
  
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
extract_year.fn <- function(raster.in = list.met[[1]],
                            year.in = 2005,
                            dataset = c('20thC_ReanV2c', 'ncep.reanalysis.derived', 'NARR')){
  
  # default to 20th cent reanalysis
  if(length(dataset) > 1){
    dataset <- dataset[1]
    print(paste('No dataset specified, defaulting to', dataset))
  }
  
  # name months 1:12 for extracting from raster
  names.months <- paste0(year.in, '-',
                         formatC(1:12, width = 2, flag = '0'), '-', '01')
  
  # extract monthly dates using function from hyspdisp
  raster.sub <- subset_nc_date(hpbl_brick = raster.in,
                               vardate = names.months)
  
  # take annual mean
  raster.sub.mean <- stackApply(raster.sub, indices = rep( 1, 12), fun = mean)
  
  # NARR dataset requires rotating
  if( dataset != 'NARR')
    raster.sub.mean <- rotate(raster.sub.mean)
  
  return(raster.sub.mean)
}


extract_month.fn <- function(
  raster.in = list.met[[1]],
  year.in = 2005,
  month.in = 1,
  dataset = c('20thC_ReanV2c', 'ncep.reanalysis.derived', 'NARR')
){
  
  # default to 20th cent reanalysis
  if(length(dataset) > 1){
    dataset <- dataset[1]
    print(paste('No dataset specified, defaulting to', dataset))
  }
  
  # name months 1:12 for extracting from raster
  names.months <- paste0(year.in, '-',
                         formatC(1:12, width = 2, flag = '0'), '-', '01')
  
  # extract monthly dates using function from hyspdisp
  raster.sub <- subset_nc_date(hpbl_brick = raster.in,
                               vardate = names.months[month.in])
  
  # NARR dataset requires rotating
  if( dataset != 'NARR')
    raster.sub <- rotate(raster.sub)
  
  return(raster.sub)
}


# trim data over US, combine into data.table, create spatial object
usa.functioner <- function(year.in = 2005,
                           list.met,
                           month.in = NULL,
                           dataset = c('20thC_ReanV2c', 'ncep.reanalysis.derived', 'NARR'),
                           return.usa.mask = F,
                           return.usa.sub = F){
  
  # extract year
  if (is.null( month.in)) {
    mets <- brick(lapply(list.met,
                       extract_year.fn,
                       year.in = year.in,
                       dataset = dataset))
  } else {
    mets <- brick(lapply(list.met,
                       extract_month.fn,
                       year.in = year.in,
                       month.in = month.in,
                       dataset = dataset))
    
  }

  
  crs.usa <- crs("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +a=6370000 +b=6370000")
  
  # convert temp to celcius
  mets$temp <- mets$temp - 273.15
  
  # calculate windspeed
  # calculate meteorology wind angle (0 is north wind)
  # http://weatherclasses.com/uploads/3/6/2/3/36231461/computing_wind_direction_and_speed_from_u_and_v.pdf
  
  # if('uwnd' %in% names(list.met) & 'vwnd' %in% names(list.met)){
  #   mets$wspd <- sqrt(mets$uwnd ^ 2 + mets$vwnd ^ 2)
  #   mets$phi <- atan2(mets$uwnd, mets$vwnd) * 180 / pi + 180
  # }
  # 
  # download USA polygon from rnaturalearth
  usa <- rnaturalearth::ne_countries(scale = 110, type = "countries", country = "United States of America", 
                                     geounit = NULL, sovereignty = NULL,
                                     returnclass = c("sp"))
  usa.sub <- disaggregate(usa)[6,]
  usa.sub <- spTransform(usa.sub, CRSobj = proj4string(mets))
  
  if(return.usa.mask){
    usa.sub.p <- spTransform(usa.sub, CRSobj = crs.usa)
    usa.sub.sf <- data.table(sf::st_as_sf(usa.sub.p))
    return(usa.sub.sf)
  }
  
  if(return.usa.sub){
    # crop to USA
    mets.usa <- crop(mask(mets, usa.sub), usa.sub)
    
    # reproject
    mets.usa <- projectRaster(mets.usa, crs = crs.usa)
    
    # convert rasters to sf - differences
    mets.usa.sp <- rasterToPolygons(mets.usa)
    
    # convert rasters to sf - annual
    mets.usa.sf <- data.table(sf::st_as_sf(mets.usa.sp))[, year := year.in]
    
    # merge with coordinates - annual
    coords <- sf::st_coordinates(sf::st_centroid(mets.usa.sf$geometry))
    mets.usa.sf <- cbind(mets.usa.sf, coords)
    
    # return sf object
    return(mets.usa.sf)
  }
  
  else{
    mets <- projectRaster(mets, crs = crs.usa)
    return(mets)
  }
}

#======================================================================#
## download monthly met data
#======================================================================#

# see all variables at
# https://psl.noaa.gov/data/gridded/data.narr.monolevel.html

# define the layer names, do the actual downloading
Sys.setenv(TZ='UTC')
layer.names <- c("air.2m.mon.mean.nc", # temperature at 2m
                 "air.sfc.mon.mean.nc", # surface temp (NEW)
                 "apcp.mon.mean.nc", # total precip,
                 "acpcp.mon.mean.nc", # acc. convective precip (NEW)
                 "tcdc.mon.mean.nc", # total cloud cover (NEW)
                 "dswrf.mon.mean.nc", # Down Short Rads Flux (NEW)
                 "hpbl.mon.mean.nc", # Planet boundary layer height (NEW)
                 "rhum.2m.mon.mean.nc", # relative humidity
                 "vwnd.10m.mon.mean.nc", # v (north-south) wind component
                 "uwnd.10m.mon.mean.nc") # u (east-west) wind component
                  
# names(layer.names) <- c("temp", "apcp", "rhum", "vwnd", "uwnd")
names(layer.names) = c(
  "temp", "stemp", "apcp", "cpcp", "tcdc",
  "dswrf", "hpbl", "rhum", "vwnd", "uwnd"
)
# names(layer.names) = c("temp", "apcp", "cpcp", "rhum", "vwnd", "uwnd")

# do the data downloading
# choose NARR for best spatial resolution  over the US
# monthly data
list.met <- lapply(layer.names,
                   downloader.fn,
                   dataset = 'NARR')

# get all data and store in binary format
years = 2000:2015
months = 1:12
controls = list()

for (y in years) {
  for (m in months) {
    ym = sprintf("%s%02d", y, m)
    controls[[ym]] = usa.functioner(y, list.met, m, dataset = 'NARR')
    print(paste("finished", ym))
  }
}

saveRDS(controls, "data/met_data.rds")
