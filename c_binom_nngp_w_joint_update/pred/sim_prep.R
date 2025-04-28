rm(list=ls())
library(RANN)

coords <- read.table("../sim_data/coords.mod")
coords.0 <- read.table("../sim_data/coords.ho")

n <- nrow(coords)
m <- 15

nn.indx <- nn2(coords, coords.0, k = m)$nn.idx

write.table(nn.indx-1, "nn.indx.0", row.names=F, col.names=F, sep="\t")
