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
real sexthal;

real agethalach;
real ageca;

real cpexang;
real cplabel;

real thalachexang;
real thalachlabel;
real thalach0;

real exangthal;
real exang0;

real oldpeaklabel;
real oldpeak0;

real slopethalach;
real slopeoldpeak;

real ca0;

real thallabel;
real thal0;

real labelca;
real label0;

}

transformed parameters {
}

model {
sexthal        ~ normal(0, 1);

agethalach        ~ normal(0, 1);
ageca        ~ normal(0, 1);

cpexang        ~ normal(0, 1);
cplabel        ~ normal(0, 1);

thalachexang        ~ normal(0, 1);
thalachlabel        ~ normal(0, 1);
thalach0        ~ normal(0, 1);

exangthal        ~ normal(0, 1);
exang0        ~ normal(0, 1);

oldpeaklabel        ~ normal(0, 1);
oldpeak0        ~ normal(0, 1);

slopethalach        ~ normal(0, 1);
slopeoldpeak        ~ normal(0, 1);

ca0        ~ normal(0, 1);

thallabel        ~ normal(0, 1);
thal0        ~ normal(0, 1);

labelca        ~ normal(0, 1);
label0        ~ normal(0, 1);

for(ind in 1:N){thalach[ind] ~ poisson(exp((agethalach * age[ind])  +(slopethalach * slope[ind])  + thalach0));
exang[ind] ~ bernoulli_logit((cpexang * cp[ind])  +(thalachexang * thalach[ind])  + exang0);
oldpeak[ind] ~ poisson(exp((slopeoldpeak * slope[ind])  + oldpeak0));
thal[ind] ~ poisson(exp((sexthal * sex[ind])  +(exangthal * exang[ind])  + thal0));
label[ind] ~ bernoulli_logit((cplabel * cp[ind])  +(thalachlabel * thalach[ind])  +(oldpeaklabel * oldpeak[ind])  +(thallabel * thal[ind])  + label0);
ca[ind] ~ poisson(exp((ageca * age[ind])  +(labelca * label[ind])  + ca0));
}

}

