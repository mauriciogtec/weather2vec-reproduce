library(tidyverse)
library(sf)
library(raster)

# read health data
eh = read_csv("data/Study_dataset_2010.csv")

# read shapefile polygons
sdf = read_sf("data/tl_2010_us_county10") %>%
    mutate(FIPS=paste0(STATEFP10, COUNTYFP10)) %>%
    left_join(eh, by="FIPS") %>%
    as("Spatial")

# raster canvas
crswgs84 = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
nrow = 128
ncol = 256
xmin = -135.0
xmax = -60.0
ymin = 20.0
ymax = 52.0
ext = extent(xmin, xmax, ymin, ymax)
rast_canvas = raster(nrow=nrow, ncol= ncol, ext=ext, crs=crswgs84)

#
vars = c(
    "qd_mean_pm25", "cs_poverty",
    "cs_hispanic", "cs_black", "cs_white",
    "cs_native", "cs_asian", "cs_ed_below_highschool",
    "cs_household_income", "cs_median_house_value", "cs_total_population",
    "cs_other", "cs_area", "cs_population_density",
    "cdc_mean_bmi", "cdc_pct_cusmoker", "cdc_pct_sdsmoker",
    "cdc_pct_fmsmoker", "cdc_pct_nvsmoker", "cdc_pct_nnsmoker",
    "gmet_mean_tmmn", "gmet_mean_summer_tmmn", "gmet_mean_winter_tmmn",
    "gmet_mean_tmmx", "gmet_mean_summer_tmmx", "gmet_mean_winter_tmmx",
    "gmet_mean_rmn", "gmet_mean_summer_rmn", "gmet_mean_winter_rmn",
    "gmet_mean_rmx", "gmet_mean_summer_rmx", "gmet_mean_winter_rmx",
    "gmet_mean_sph", "gmet_mean_summer_sph", "gmet_mean_winter_sph",
    "cms_mortality_pct", "cms_white_pct", "cms_black_pct",
    "cms_others_pct", "cms_hispanic_pct", "cms_female_pct"
)

rasters = map(
    vars,
    ~ rasterize(
        sdf, rast_canvas,
        field = sdf@data[, .x],
        update = TRUE,
        updateValue = "NA"
    )
)
names(rasters) = vars
br = brick(rasters)

dir.create("data/medicare")
writeRaster(
    br,
    filename='data/medicare/medicare.tif',
    format="GTiff",
    overwrite=TRUE,
    options=c("INTERLEAVE=BAND", "COMPRESS=LZW")
)

write_lines(vars, "data/medicare/medicare_names.txt")


# make radom masks with 10% holdout by pre-masking counties
dir.create("data/medicare/holdout_masks/")
nreps = 100
test_frac = 0.05
poly =  read_sf("data/tl_2010_us_county10") %>% as("Spatial")
n = length(poly)

for (i in 1:nreps) {
    tmp = sdf
    ixs = which(tmp@data$cms_mortality_pct > 0)
    which_test = (runif(length(ixs)) < test_frac)
    tmp@data$test = 0
    tmp@data$test[ixs[which_test]] = 1

    test = rasterize(
        tmp, rast_canvas,
        field = tmp@data$test,
        update = TRUE,
        updateValue = "NA"
    )
    writeRaster(test, sprintf("data/medicare/holdout_masks/%03d.tif", i - 1), overwrite=TRUE)
}
