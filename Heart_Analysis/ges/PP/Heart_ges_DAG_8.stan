data{
int<lower = 0> N;
array[N] int<lower=0, upper=1>  sex;
array[N] int<lower=2, upper=7>  age;
array[N] int<lower=1, upper=4>  cp;
array[N] int<lower=71, upper=202>  thalach;
array[N] int<lower=0, upper=1>  exang;
array[N] int<lower=0, upper=6>  oldpeak;
array[N] int<lower=1, upper=3>  slope;
array[N] int<lower=0, upper=3>  ca;
array[N] int<lower=3, upper=7>  thal;
array[N] int<lower=0, upper=1>  label;
}

transformed data {
}

parameters {
real sex0;

real agesex;
real ageca;
real age0;

real cpthalach;
real cpexang;
real cp0;

real thalachage;
real thalachexang;
real thalach0;

real exang0;

real oldpeak0;

real slopethalach;
real slopeoldpeak;
real slopethal;

real ca0;

real thalsex;
real thalexang;
real thal0;

real labelcp;
real labelthalach;
real labeloldpeak;
real labelca;
real labelthal;

}

transformed parameters {
}

model {
sex0        ~ normal(0, 1);

agesex        ~ normal(0, 1);
ageca        ~ normal(0, 1);
age0        ~ normal(0, 1);

cpthalach        ~ normal(0, 1);
cpexang        ~ normal(0, 1);
cp0        ~ normal(0, 1);

thalachage        ~ normal(0, 1);
thalachexang        ~ normal(0, 1);
thalach0        ~ normal(0, 1);

exang0        ~ normal(0, 1);

oldpeak0        ~ normal(0, 1);

slopethalach        ~ normal(0, 1);
slopeoldpeak        ~ normal(0, 1);
slopethal        ~ normal(0, 1);

ca0        ~ normal(0, 1);

thalsex        ~ normal(0, 1);
thalexang        ~ normal(0, 1);
thal0        ~ normal(0, 1);

labelcp        ~ normal(0, 1);
labelthalach        ~ normal(0, 1);
labeloldpeak        ~ normal(0, 1);
labelca        ~ normal(0, 1);
labelthal        ~ normal(0, 1);

for(ind in 1:N){cp[ind] ~ poisson(exp((labelcp * label[ind])  + cp0));
thalach[ind] ~ poisson(exp((cpthalach * cp[ind])  +(slopethalach * slope[ind])  +(labelthalach * label[ind])  + thalach0));
oldpeak[ind] ~ poisson(exp((slopeoldpeak * slope[ind])  +(labeloldpeak * label[ind])  + oldpeak0));
thal[ind] ~ poisson(exp((slopethal * slope[ind])  +(labelthal * label[ind])  + thal0));
age[ind] ~ poisson(exp((thalachage * thalach[ind])  + age0));
exang[ind] ~ bernoulli_logit((cpexang * cp[ind])  +(thalachexang * thalach[ind])  +(thalexang * thal[ind])  + exang0);
ca[ind] ~ poisson(exp((ageca * age[ind])  +(labelca * label[ind])  + ca0));
sex[ind] ~ bernoulli_logit((agesex * age[ind])  +(thalsex * thal[ind])  + sex0);
}

}

