#!/bin/bash
Rscript --vanilla preprocessing/download_weather.R
Rscript --vanilla preprocessing/process_weather.R
Rscript --vanilla preprocessing/process_so4.R
Rscript --vanilla preprocessing/process_pp.R
Rscript --vanilla preprocessing/process_medicare.R
