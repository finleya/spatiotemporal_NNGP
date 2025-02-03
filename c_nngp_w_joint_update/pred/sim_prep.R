rm(list=ls())
library(RANN)

coords <- read.table("../../sim_data/coords")
coords.0 <- read.table("../../sim_data/coords")

n <- nrow(coords)
m <- 10

nn.indx <- nn2(coords, coords.0, k = m)$nn.idx

write.table(nn.indx-1, "nn.indx.0", row.names=F, col.names=F, sep="\t")
