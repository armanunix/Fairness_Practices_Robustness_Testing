data{
int<lower = 0> N;
array[N] int<lower=0, upper=1>  sex;
array[N] int<lower=0, upper=1>  race;
array[N] int<lower=0, upper=2>  age;
array[N] int<lower=0, upper=38>  p;
array[N] int<lower=0, upper=20>  j;
array[N] int<lower=0, upper=10>  d;
array[N] int<lower=0, upper=1>  y;
}

transformed data {
}

parameters {
real sexrace;
real sexp;
real sexd;

real racey;
real race0;

real agey;
real age0;

real prace;
real page;
real pj;
real py;
real p0;

real jrace;
real jy;
real j0;

real dp;
real dy;
real d0;

real y0;

}

transformed parameters {
}

model {
sexrace        ~ normal(0, 1);
sexp        ~ normal(0, 1);
sexd        ~ normal(0, 1);

racey        ~ normal(0, 1);
race0        ~ normal(0, 1);

agey        ~ normal(0, 1);
age0        ~ normal(0, 1);

prace        ~ normal(0, 1);
page        ~ normal(0, 1);
pj        ~ normal(0, 1);
py        ~ normal(0, 1);
p0        ~ normal(0, 1);

jrace        ~ normal(0, 1);
jy        ~ normal(0, 1);
j0        ~ normal(0, 1);

dp        ~ normal(0, 1);
dy        ~ normal(0, 1);
d0        ~ normal(0, 1);

y0        ~ normal(0, 1);

for(ind in 1:N){d[ind] ~ poisson(exp((sexd * sex[ind])  + d0));
p[ind] ~ poisson(exp((sexp * sex[ind])  +(dp * d[ind])  + p0));
j[ind] ~ poisson(exp((pj * p[ind])  + j0));
race[ind] ~ bernoulli_logit((sexrace * sex[ind])  +(prace * p[ind])  +(jrace * j[ind])  + race0);
age[ind] ~ poisson(exp((page * p[ind])  + age0));
y[ind] ~ bernoulli_logit((racey * race[ind])  +(agey * age[ind])  +(py * p[ind])  +(jy * j[ind])  +(dy * d[ind])  + y0);
}

}

