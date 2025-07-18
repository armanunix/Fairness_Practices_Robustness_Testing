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
real race0;

real sexrace;
real sexLSAT;
real sexUGPA;

real LSATrace;
real LSATUGPA;
real LSAT0;

real UGPA0;

real first_pfrace;
real first_pfLSAT;
real first_pfUGPA;

real<lower=0> sigma_h_Sq;
}

transformed parameters {
real<lower=0> sigma_h;
sigma_h = sqrt(sigma_h_Sq);
}

model {
raceUGPA        ~ normal(0, 1);
race0        ~ normal(0, 1);

sexrace        ~ normal(0, 1);
sexLSAT        ~ normal(0, 1);
sexUGPA        ~ normal(0, 1);

LSATrace        ~ normal(0, 1);
LSATUGPA        ~ normal(0, 1);
LSAT0        ~ normal(0, 1);

UGPA0        ~ normal(0, 1);

first_pfrace        ~ normal(0, 1);
first_pfLSAT        ~ normal(0, 1);
first_pfUGPA        ~ normal(0, 1);

sigma_h_Sq ~ inv_gamma(1, 1);

for(ind in 1:N){LSAT[ind] ~ normal((sexLSAT * sex[ind])  +(first_pfLSAT * first_pf[ind])  + LSAT0, sigma_h);
race[ind] ~ poisson(exp((sexrace * sex[ind])  +(LSATrace * LSAT[ind])  +(first_pfrace * first_pf[ind])  + race0));
UGPA[ind] ~ normal((raceUGPA * race[ind])  +(sexUGPA * sex[ind])  +(LSATUGPA * LSAT[ind])  +(first_pfUGPA * first_pf[ind])  + UGPA0, sigma_h);
}

}

