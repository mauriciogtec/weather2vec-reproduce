library(tidyverse)
library(data.table)
library(raster)
library(proj4)
library(rgdal)

# -- Plant data
min_lon = -80.55
min_lat = 39.65
max_lon = -75.25
max_lat = 42.05

target_years = 2013
target_month = 1
n_lags = 6
n_obs = 12

dat = fread("./data/AMPD_Unit_with_Sulfur_Content_and_Regulations_with_Facility_Attributes.csv")
dat = dat[ ,-1]  # drop first column which contains only row number

dat[ ,Fac.ID := dat$Facility.ID..ORISPL.]
dat[ ,uID := paste(dat$Fac.ID, dat$Unit.ID, sep = "_")]
dat[, year_month := paste(Year, Month, sep="_")]
setkeyv(dat, c("uID", "Year", "Month"))
setorderv(dat, c("uID", "Year", "Month"))
dim(dat)  # there are some duplicates
dat = unique(dat)  # remove duplicates
dim(dat)

df_out = dat %>%
  as_tibble  %>%
  rename(fid = Fac.ID,
         so2_tons = SO2..tons.,
         lat = Facility.Latitude.x,
         lon = Facility.Longitude.x,
         month = Month,
         year = Year,
         fuel_type = Fuel.Type..Primary..x,
         state = State.x) #%>%
  # filter(fuel_type == "Coal")

expand_rle = function(rle_res) {
  out = c()
  lens = rle_res$lengths
  vals = rle_res$values
  k = 1
  for (i in 1:length(lens)) {
    to_fill = ifelse(vals[i] == TRUE, lens[i], 0)
    for (j in 1:lens[i]) {
      out[k] = to_fill
      k = k + 1
    }
  }
  return (out)
}

operating_time = df_out %>% 
  group_by(fid, year, month, lat, lon, fuel_type) %>%
  summarise(
    operating_time=sum(Operating.Time, na.rm=TRUE),
    heat_input=sum(Heat.Input..MMBtu., na.rm=TRUE),
    so2_tons=sum(so2_tons, na.rm=TRUE)
  , .groups="drop") %>% 
  complete(nesting(fid, lat, lon), year, month, fill=list(operating_time=NA)) %>% 
  group_by(fid) %>% 
  mutate(na_streak = expand_rle(rle(is.na(operating_time)))) %>% 
  mutate(op_is_zero = operating_time == 0) %>% 
  replace_na(list(op_is_zero=-1)) %>%
  mutate(zero_streak = expand_rle(rle(op_is_zero))) %>% 
  ungroup()
write_csv(operating_time, "./data/so2_data_full.csv")

pp_sizes = operating_time %>% 
  group_by(fid, lon, lat, fuel_type) %>% 
  summarize(
    so2_total=sum(so2_tons, na.rm=TRUE),
    miss_total=sum(na_streak) + sum(zero_streak),
    .groups="drop"
  )
write_csv(pp_sizes, "./data/pp_sizes.csv")


# df_out2 = df_out %>%
#   dplyr::select(fid, month, year, so2_tons, lat, lon) %>%
#   na.omit() %>%
#   group_by(fid, month, year) %>%
#   summarise_all(sum)
# write_csv(df_out, "./data/so2_data.csv")

# 
power_plant_info = operating_time %>%
  ungroup() %>%
  dplyr::select(fid, lat, lon, fuel_type) %>%
  distinct(fid, .keep_all = TRUE)
# 
write_csv(power_plant_info, "./data/power_plant_info.csv")
