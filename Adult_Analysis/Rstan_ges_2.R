library('rstan')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

raw_data <- read.csv('../subjects/datasets/adult_org-Copy1.csv')
n_samples = nrow(raw_data)
data_pp <- list(
    N = n_samples,
    age = raw_data$age,
    w = raw_data$w,
    e = raw_data$e,
    m = raw_data$m,    
    o = raw_data$o,
    r = raw_data$r,
    race = raw_data$race,
    gender = raw_data$gender,    
    hr = raw_data$hr,
    n = raw_data$n,  
    y = raw_data$y)
file_num <- 2

file_name <- sprintf("./ges/PP/Adult_ges_DAG_%d.stan", file_num)
print(file_name)
fit <- stan(file = file_name,iter = 1000, data = data_pp, chains = 1)
res = as.data.frame(fit)
csv_filenme <- sprintf("./ges/PP/Adult_ges_pp_%d.csv", file_num)
write.csv(res,csv_filenme)