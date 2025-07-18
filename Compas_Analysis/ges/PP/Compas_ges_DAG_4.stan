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
real sex0;

real racesex;
real racep;
real racej;
real racey;

real agep;
real agej;
real age0;

real psex;
real pj;
real p0;

real j0;

real dsex;
real dp;
real dj;
real d0;

real yage;
real yp;
real yd;
real y0;

}

transformed parameters {
}

model {
sex0        ~ normal(0, 1);

racesex        ~ normal(0, 1);
racep        ~ normal(0, 1);
racej        ~ normal(0, 1);
racey        ~ normal(0, 1);

agep        ~ normal(0, 1);
agej        ~ normal(0, 1);
age0        ~ normal(0, 1);

psex        ~ normal(0, 1);
pj        ~ normal(0, 1);
p0        ~ normal(0, 1);

j0        ~ normal(0, 1);

dsex        ~ normal(0, 1);
dp        ~ normal(0, 1);
dj        ~ normal(0, 1);
d0        ~ normal(0, 1);

yage        ~ normal(0, 1);
yp        ~ normal(0, 1);
yd        ~ normal(0, 1);
y0        ~ normal(0, 1);

for(ind in 1:N){y[ind] ~ bernoulli_logit((racey * race[ind])  + y0);
age[ind] ~ poisson(exp((yage * y[ind])  + age0));
d[ind] ~ poisson(exp((yd * y[ind])  + d0));
p[ind] ~ poisson(exp((racep * race[ind])  +(agep * age[ind])  +(dp * d[ind])  +(yp * y[ind])  + p0));
j[ind] ~ poisson(exp((racej * race[ind])  +(agej * age[ind])  +(pj * p[ind])  +(dj * d[ind])  + j0));
sex[ind] ~ bernoulli_logit((racesex * race[ind])  +(psex * p[ind])  +(dsex * d[ind])  + sex0);
}

}

