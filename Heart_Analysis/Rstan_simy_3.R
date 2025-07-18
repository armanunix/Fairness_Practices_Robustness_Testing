library('rstan')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

raw_data <- read.csv('../subjects/datasets/heart_processed_1')
n_samples = nrow(raw_data)
data_pp <- list(

    N = n_samples,
    age = raw_data$age,
    sex = raw_data$sex,
    cp = raw_data$cp,
    thalach = raw_data$thalach,    
    exang = raw_data$exang,
    oldpeak = raw_data$oldpeak,
    slope = raw_data$slope,
    ca = raw_data$ca,    
    thal = raw_data$thal,
    label = raw_data$label)
file_num <- 3

file_name <- sprintf("./simy/PP/Heart_simy_DAG_%d.stan", file_num)
print(file_name)
fit <- stan(file = file_name,iter = 1000, data = data_pp, chains = 1)
res = as.data.frame(fit)
csv_filenme <- sprintf("./simy/PP/Heart_simy_pp_%d.csv", file_num)
write.csv(res,csv_filenme)