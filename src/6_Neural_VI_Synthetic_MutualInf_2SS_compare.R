## Find a mask for the distance matrix to be used in the summary statistic
D_mask_tf <- (D_tf < (mean(diff(s1))*1.1)) %>%
              tf$where(1L, 0L)  %>%
              tf$cast("float64")
test_images <- readRDS("data/test_images.rds")
test_lscales <- readRDS("data/test_lscales.rds")
Z <- test_images
Z_long <- tf$reshape(Z, c(-1L, ngrid_squared, 1L))
Z_long_t <- tf$linalg$matrix_transpose(Z_long)
XX <- tf$linalg$matmul(Z_long, Z_long_t)
XXX <- tf$multiply(XX, D_mask_tf)
S1 <- tf$reduce_sum(tf$linalg$diag_part(XXX), c(1L), keepdims = TRUE) %>% tf$squeeze()
S2 <- tf$reduce_sum(XXX, c(1L,2L), keepdims = TRUE) %>% tf$squeeze() - S1
png("temp.png")
plot(test_lscales, S2/S1)
dev.off()
