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
real racesex;
real raceLSAT;
real raceUGPA;
real race0;

real sexLSAT;
real sexUGPA;
real sex0;

real LSAT0;

real UGPALSAT;
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
racesex        ~ normal(0, 1);
raceLSAT        ~ normal(0, 1);
raceUGPA        ~ normal(0, 1);
race0        ~ normal(0, 1);

sexLSAT        ~ normal(0, 1);
sexUGPA        ~ normal(0, 1);
sex0        ~ normal(0, 1);

LSAT0        ~ normal(0, 1);

UGPALSAT        ~ normal(0, 1);
UGPA0        ~ normal(0, 1);

first_pfrace        ~ normal(0, 1);
first_pfLSAT        ~ normal(0, 1);
first_pfUGPA        ~ normal(0, 1);

sigma_h_Sq ~ inv_gamma(1, 1);

for(ind in 1:N){race[ind] ~ poisson(exp((first_pfrace * first_pf[ind])  + race0));
sex[ind] ~ bernoulli_logit((racesex * race[ind])  + sex0);
UGPA[ind] ~ normal((raceUGPA * race[ind])  +(sexUGPA * sex[ind])  +(first_pfUGPA * first_pf[ind])  + UGPA0, sigma_h);
LSAT[ind] ~ normal((raceLSAT * race[ind])  +(sexLSAT * sex[ind])  +(UGPALSAT * UGPA[ind])  +(first_pfLSAT * first_pf[ind])  + LSAT0, sigma_h);
}

}

