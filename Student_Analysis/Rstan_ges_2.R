library('rstan')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

raw_data <- read.csv('../subjects/datasets/students-processed_2')
n_samples = nrow(raw_data)
data_pp <- list(
    N = n_samples,
    age = raw_data$age,
    sex = raw_data$sex,
    school = raw_data$school,
    Medu = raw_data$Medu,    
    Fedu = raw_data$Fedu,
    reason = raw_data$reason,
    guardian = raw_data$guardian,
    studytime = raw_data$studytime,    
    failures = raw_data$failures,
    higher = raw_data$higher,
    romantic = raw_data$romantic,
    goout = raw_data$goout,    
    Dalc = raw_data$Dalc,
    absences = raw_data$absences,
    G1 = raw_data$G1,    
    G2 = raw_data$G2,
    G3 = raw_data$G3)
file_num <- 2

file_name <- sprintf("./ges/PP/Student_ges_DAG_%d.stan", file_num)
print(file_name)
fit <- stan(file = file_name,iter = 1000, data = data_pp, chains = 1)
res = as.data.frame(fit)
csv_filenme <- sprintf("./ges/PP/Student_ges_pp_%d.csv", file_num)
write.csv(res,csv_filenme)