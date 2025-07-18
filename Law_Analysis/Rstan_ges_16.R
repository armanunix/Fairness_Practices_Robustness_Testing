library('rstan')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)




raw_data <- read.csv('../subjects/datasets/law.csv')
n_samples = nrow(raw_data)
data_pp <- list(
    N = n_samples,
    LSAT = raw_data$LSAT,
    race = raw_data$race,
    sex = raw_data$sex,
    UGPA = raw_data$UGPA,  
    first_pf = raw_data$first_pf)
file_num <- 16

file_name <- sprintf("./ges/PP/Law_ges_DAG_%d.stan", file_num)
print(file_name)
fit <- stan(file = file_name,iter = 1000, data = data_pp, chains = 1)
res = as.data.frame(fit)
csv_filenme <- sprintf("./ges/PP/Law_ges_pp_%d.csv", file_num)
write.csv(res,csv_filenme)