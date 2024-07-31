install.packages(c("egg", "ggplot2", "JuliaConnectoR", "fields", "dplyr",
                  "reticulate", "tensorflow", "SpatialExtremes",
                  "geoR", "keras", "grid", "gtable", "gridExtra",
                  "xtable", "optparse", "remotes", "latex2exp"),
                 repos='http://cran.us.r-project.org')

library("remotes")
remotes::install_github("https://github.com/msainsburydale/NeuralEstimators")
