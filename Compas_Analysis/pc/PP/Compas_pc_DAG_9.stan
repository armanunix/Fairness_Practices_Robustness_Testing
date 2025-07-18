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
real sexj;
real sexy;
real sex0;

real racej;
real race0;

real agerace;
real agep;
real agej;
real agey;

real prace;
real pj;
real p0;

real j0;

real dsex;
real dp;
real dj;
real dy;

real yrace;
real yp;
real yj;
real y0;

}

transformed parameters {
}

model {
sexrace        ~ normal(0, 1);
sexp        ~ normal(0, 1);
sexj        ~ normal(0, 1);
sexy        ~ normal(0, 1);
sex0        ~ normal(0, 1);

racej        ~ normal(0, 1);
race0        ~ normal(0, 1);

agerace        ~ normal(0, 1);
agep        ~ normal(0, 1);
agej        ~ normal(0, 1);
agey        ~ normal(0, 1);

prace        ~ normal(0, 1);
pj        ~ normal(0, 1);
p0        ~ normal(0, 1);

j0        ~ normal(0, 1);

dsex        ~ normal(0, 1);
dp        ~ normal(0, 1);
dj        ~ normal(0, 1);
dy        ~ normal(0, 1);

yrace        ~ normal(0, 1);
yp        ~ normal(0, 1);
yj        ~ normal(0, 1);
y0        ~ normal(0, 1);

for(ind in 1:N){sex[ind] ~ bernoulli_logit((dsex * d[ind])  + sex0);
y[ind] ~ bernoulli_logit((sexy * sex[ind])  +(agey * age[ind])  +(dy * d[ind])  + y0);
p[ind] ~ poisson(exp((sexp * sex[ind])  +(agep * age[ind])  +(dp * d[ind])  +(yp * y[ind])  + p0));
race[ind] ~ bernoulli_logit((sexrace * sex[ind])  +(agerace * age[ind])  +(prace * p[ind])  +(yrace * y[ind])  + race0);
j[ind] ~ poisson(exp((sexj * sex[ind])  +(racej * race[ind])  +(agej * age[ind])  +(pj * p[ind])  +(dj * d[ind])  +(yj * y[ind])  + j0));
}

}

