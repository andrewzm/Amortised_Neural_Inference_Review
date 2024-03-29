install.packages(c("ggplot2", "JuliaConnectoR", "fields", "dplyr",
                  "reticulate", "tensorflow", "SpatialExtremes",
                  "geoR", "keras", "grid", "gtable", "gridExtra", 
                  "xtable", "optparse", "remotes"), repos='http://cran.us.r-project.org')

library("remotes")
remotes::install_github("https://github.com/msainsburydale/NeuralEstimators")