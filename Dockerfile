FROM rocker/geospatial:3.6.3

RUN R -e "install.packages('MatchIt', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('Matching', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('corrplot', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('fields', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('remotes', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('optmatch', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('splines2', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('cowplot', repos='https://cloud.r-project.org/')"
RUN R -e "install.packages('languageserver', repos='https://cloud.r-project.org/')"
RUN R -e "remotes::install_github('cran/zipcode')"
RUN R -e "remotes::install_github('czigler/arepa')"
RUN R -e "remotes::install_github('gpapadog/DAPSm')"

WORKDIR /workspace