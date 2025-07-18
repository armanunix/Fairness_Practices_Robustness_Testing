data{
int<lower = 0> N;
array[N] int<lower=0.0, upper=7.0>  race;
array[N] int<lower=0.0, upper=1.0>  sex;
array[N] real<lower=11.0, upper=48.0>  LSAT;
array[N] real<lower=0.0, upper=4.2>  UGPA;
array[N] int<lower=0.0, upper=1.0>  first_pf;
}

transformed data {
}

parameters {
real raceUGPA;
real racefirst_pf;
real race0;

real sexrace;
real sexUGPA;
real sex0;

real LSATrace;
real LSATsex;
real LSATUGPA;
real LSATfirst_pf;

real UGPA0;

real first_pfUGPA;
real first_pf0;

real<lower=0> sigma_h_Sq;
}

transformed parameters {
real<lower=0> sigma_h;
sigma_h = sqrt(sigma_h_Sq);
}

model {
raceUGPA        ~ normal(0, 1);
racefirst_pf        ~ normal(0, 1);
race0        ~ normal(0, 1);

sexrace        ~ normal(0, 1);
sexUGPA        ~ normal(0, 1);
sex0        ~ normal(0, 1);

LSATrace        ~ normal(0, 1);
LSATsex        ~ normal(0, 1);
LSATUGPA        ~ normal(0, 1);
LSATfirst_pf        ~ normal(0, 1);

UGPA0        ~ normal(0, 1);

first_pfUGPA        ~ normal(0, 1);
first_pf0        ~ normal(0, 1);

sigma_h_Sq ~ inv_gamma(1, 1);

for(ind in 1:N){sex[ind] ~ bernoulli_logit((LSATsex * LSAT[ind])  + sex0);
race[ind] ~ poisson(exp((sexrace * sex[ind])  +(LSATrace * LSAT[ind])  + race0));
first_pf[ind] ~ bernoulli_logit((racefirst_pf * race[ind])  +(LSATfirst_pf * LSAT[ind])  + first_pf0);
UGPA[ind] ~ normal((raceUGPA * race[ind])  +(sexUGPA * sex[ind])  +(LSATUGPA * LSAT[ind])  +(first_pfUGPA * first_pf[ind])  + UGPA0, sigma_h);
}

}

