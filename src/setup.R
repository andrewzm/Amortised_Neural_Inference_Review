# Author: Andrew Zammit-Mangion, azm (at) uow.edu.au
# Date: 2024-02-15
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

suppressMessages({
library("ggplot2")
library("NeuralEstimators")
library("JuliaConnectoR")
library("dplyr")
})
library("optparse")

option_list <- list(
  make_option("--statmodel", type="character", default="GP", metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
statmodel       <- parse_args(opt_parser)$statmodel

config_setup <- function(statmodel, method_name) {
  settings <- list()
  if(statmodel == "GP") {
    settings$fname_train_data <- "data/train_images.rds"
    settings$fname_train_params <- "data/train_params.rds"
    settings$fname_val_data <- "data/val_images.rds"
    settings$fname_val_params <- "data/val_params.rds"
    settings$fname_test_data <- "data/test_images.rds"
    settings$fname_test_params <- "data/test_params.rds"
    settings$fname_micro_test_data <- "data/micro_test_images.rds"
    settings$fname_micro_test_params <- "data/micro_test_params.rds"
    settings$ckpt_path <- paste0("ckpts/", method_name, "/")
    settings$output_path <- paste0("output/", method_name, "_test.rds")
    settings$output_micro_path <- paste0("output/", method_name, "_micro_test.rds")
    settings$support <- 0.6   # upper bound of support
    settings$support_min <- 0 # lower bound of support
    settings$num_params <- 1L
  } else if(statmodel == "MSP") {
    settings$fname_train_data <- "data/train_MSP_images.rds"
    settings$fname_train_params <- "data/train_MSP_params.rds"
    settings$fname_val_data <- "data/val_MSP_images.rds"
    settings$fname_val_params <- "data/val_MSP_params.rds"
    settings$fname_test_data <- "data/test_MSP_images.rds"
    settings$fname_test_params <- "data/test_MSP_params.rds"
    settings$fname_micro_test_data <- "data/micro_MSP_test_images.rds"
    settings$fname_micro_test_params <- "data/micro_MSP_test_params.rds"
    settings$ckpt_path <- paste0("ckpts/", method_name, "_MSP/")
    settings$output_path <- paste0("output/", method_name, "_MSP_test.rds")
    settings$output_micro_path <- paste0("output/", method_name, "_MSP_micro_test.rds")
    settings$support <- c(0.6, 3)     # upper bound of support
    settings$support_min <- c(0, 0.5) # lower bound of support
    settings$num_params <- 2L
  } else {
    stop("Method must be GP or MSP")
  }
  settings
}
