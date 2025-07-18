library('rstan')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# raw_data <- read.csv('../subjects/datasets/bank')
# n_samples = nrow(raw_data)
# data_pp <- list(
#     N = n_samples,
#     age = raw_data$age,
#     b = raw_data$b,
#     c = raw_data$c,
#     d = raw_data$d,    
#     e = raw_data$e,
#     f = raw_data$f,
#     g = raw_data$g,
#     h = raw_data$h,    
#     i = raw_data$i,
#     j = raw_data$j,
#     k = raw_data$k,
#     l = raw_data$l,
#     m = raw_data$m,
#     n = raw_data$n,
#     o = raw_data$o,
#     p = raw_data$p,
#     label = raw_data$label)
# fit <- stan(file = '../Bank_Analysis/ges/Bank_ges_2.stan',iter = 1000, data = data_pp, chains = 4,cores = 8)
# res = as.data.frame(fit)
# write.csv(res,'../Bank_Analysis/ges/bank_pp_GES_2.csv')



raw_data <- read.csv('../subjects/datasets/bank')

n_samples = nrow(raw_data)
data_pp <- list(
    N = n_samples,
    age = raw_data$age,
    b = raw_data$b,
    c = raw_data$c,
    d = raw_data$d,    
    e = raw_data$e,
    f = raw_data$f,
    g = raw_data$g,
    h = raw_data$h,    
    i = raw_data$i,
    j = raw_data$j,
    k = raw_data$k,
    l = raw_data$l,
    m = raw_data$m,
    n = raw_data$n,
    o = raw_data$o,
    p = raw_data$p,
    label = raw_data$label)
file_num <- 14
if (file_num%in%c(5,7,10,12)){next}
file_name <- sprintf("./ges/Bank_ges_%d.stan", file_num)
print(file_name)
fit <- stan(file = file_name,iter = 1000, data = data_pp, chains = 1)
res = as.data.frame(fit)
csv_filenme <- sprintf("./ges/bank_pp_%d.csv", file_num)
write.csv(res,csv_filenme)}