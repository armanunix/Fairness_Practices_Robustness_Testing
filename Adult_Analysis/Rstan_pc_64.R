library('rstan')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

raw_data <- read.csv('../subjects/datasets/adult_org-Copy1.csv')
#colnames(raw_data) <- c('age','w','e','m','o','r','race','gender','hr','n','y')
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
file_num <- 64

file_name <- sprintf("./pc/PP/Adult_pc_DAG_%d.stan", file_num)
print(file_name)
fit <- stan(file = file_name,iter = 1000, data = data_pp, chains = 1)
res = as.data.frame(fit)
csv_filenme <- sprintf("./pc/PP/Adult_pc_pp_%d.csv", file_num)
write.csv(res,csv_filenme)