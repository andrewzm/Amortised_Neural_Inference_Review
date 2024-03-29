install.packages(c("ggplot2", "JuliaConnectoR", "fields", "dplyr",
                  "reticulate", "tensorflow", "SpatialExtremes",
                  "geoR", "keras", "grid", "gtable", "gridExtra", 
                  "xtable", "optparse", "devtools"), repos='http://cran.us.r-project.org')

library("devtools")
devtools::install_github("https://github.com/msainsburydale/NeuralEstimators")