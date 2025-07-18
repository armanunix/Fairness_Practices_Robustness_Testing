library('rstan')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)




raw_data <- read.csv('../subjects/datasets/compas-Copy1')
n_samples = nrow(raw_data)
data_pp <- list(
    N = n_samples,
    age = raw_data$age,
    race = raw_data$race,
    sex = raw_data$sex,
    p = raw_data$p,
    j = raw_data$j,
    d = raw_data$d,    
    y = raw_data$y)
file_num <- 8

file_name <- sprintf("./simy/PP/Compas_simy_DAG_%d.stan", file_num)
print(file_name)
fit <- stan(file = file_name,iter = 1000, data = data_pp, chains = 1)
res = as.data.frame(fit)
csv_filenme <- sprintf("./simy/PP/Compas_simy_pp_%d.csv", file_num)
write.csv(res,csv_filenme)