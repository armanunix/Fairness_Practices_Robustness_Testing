data{
int<lower = 0> N;
array[N] int<lower=1.0, upper=9.0>  age;
array[N] int<lower=0.0, upper=8.0>  w;
array[N] int<lower=1.0, upper=16.0>  e;
array[N] int<lower=0.0, upper=6.0>  m;
array[N] int<lower=0.0, upper=14.0>  o;
array[N] int<lower=0.0, upper=5.0>  r;
array[N] int<lower=0.0, upper=4.0>  race;
array[N] int<lower=0.0, upper=1.0>  gender;
array[N] real<lower=-3.1814521274018306, upper=4.727311617850181>  hr;
array[N] int<lower=0.0, upper=41.0>  n;
array[N] int<lower=0.0, upper=1.0>  y;
}

transformed data {
}

parameters {
real agem;
real agerace;
real agen;
real age0;

real wrace;
real wn;
real w0;

real eage;
real ew;
real em;
real er;
real erace;
real egender;
real ehr;
real en;
real e0;

real mw;
real mrace;
real mn;
real m0;

real oage;
real ow;
real oe;
real om;
real or;
real orace;
real ogender;
real ohr;
real on;
real oy;

real rage;
real rw;
real rm;
real rrace;
real rgender;
real rhr;
real r0;

real racen;
real race0;

real genderage;
real genderw;
real genderm;
real genderrace;
real gendern;
real gender0;

real hrage;
real hrw;
real hrm;
real hrrace;
real hrgender;
real hrn;
real hr0;

real n0;

real yage;
real yw;
real ye;
real ym;
real yr;
real yrace;
real ygender;
real yhr;
real yn;
real y0;

real<lower=0> sigma_h_Sq;
}

transformed parameters {
real<lower=0> sigma_h;
sigma_h = sqrt(sigma_h_Sq);
}

model {
agem        ~ normal(0, 1);
agerace        ~ normal(0, 1);
agen        ~ normal(0, 1);
age0        ~ normal(0, 1);

wrace        ~ normal(0, 1);
wn        ~ normal(0, 1);
w0        ~ normal(0, 1);

eage        ~ normal(0, 1);
ew        ~ normal(0, 1);
em        ~ normal(0, 1);
er        ~ normal(0, 1);
erace        ~ normal(0, 1);
egender        ~ normal(0, 1);
ehr        ~ normal(0, 1);
en        ~ normal(0, 1);
e0        ~ normal(0, 1);

mw        ~ normal(0, 1);
mrace        ~ normal(0, 1);
mn        ~ normal(0, 1);
m0        ~ normal(0, 1);

oage        ~ normal(0, 1);
ow        ~ normal(0, 1);
oe        ~ normal(0, 1);
om        ~ normal(0, 1);
or        ~ normal(0, 1);
orace        ~ normal(0, 1);
ogender        ~ normal(0, 1);
ohr        ~ normal(0, 1);
on        ~ normal(0, 1);
oy        ~ normal(0, 1);

rage        ~ normal(0, 1);
rw        ~ normal(0, 1);
rm        ~ normal(0, 1);
rrace        ~ normal(0, 1);
rgender        ~ normal(0, 1);
rhr        ~ normal(0, 1);
r0        ~ normal(0, 1);

racen        ~ normal(0, 1);
race0        ~ normal(0, 1);

genderage        ~ normal(0, 1);
genderw        ~ normal(0, 1);
genderm        ~ normal(0, 1);
genderrace        ~ normal(0, 1);
gendern        ~ normal(0, 1);
gender0        ~ normal(0, 1);

hrage        ~ normal(0, 1);
hrw        ~ normal(0, 1);
hrm        ~ normal(0, 1);
hrrace        ~ normal(0, 1);
hrgender        ~ normal(0, 1);
hrn        ~ normal(0, 1);
hr0        ~ normal(0, 1);

n0        ~ normal(0, 1);

yage        ~ normal(0, 1);
yw        ~ normal(0, 1);
ye        ~ normal(0, 1);
ym        ~ normal(0, 1);
yr        ~ normal(0, 1);
yrace        ~ normal(0, 1);
ygender        ~ normal(0, 1);
yhr        ~ normal(0, 1);
yn        ~ normal(0, 1);
y0        ~ normal(0, 1);

sigma_h_Sq ~ inv_gamma(1, 1);

for(ind in 1:N){y[ind] ~ bernoulli_logit((oy * o[ind])  + y0);
e[ind] ~ poisson(exp((oe * o[ind])  +(ye * y[ind])  + e0));
r[ind] ~ poisson(exp((er * e[ind])  +(or * o[ind])  +(yr * y[ind])  + r0));
hr[ind] ~ normal((ehr * e[ind])  +(ohr * o[ind])  +(rhr * r[ind])  +(yhr * y[ind])  + hr0, sigma_h);
gender[ind] ~ bernoulli_logit((egender * e[ind])  +(ogender * o[ind])  +(rgender * r[ind])  +(hrgender * hr[ind])  +(ygender * y[ind])  + gender0);
age[ind] ~ poisson(exp((eage * e[ind])  +(oage * o[ind])  +(rage * r[ind])  +(genderage * gender[ind])  +(hrage * hr[ind])  +(yage * y[ind])  + age0));
m[ind] ~ poisson(exp((agem * age[ind])  +(em * e[ind])  +(om * o[ind])  +(rm * r[ind])  +(genderm * gender[ind])  +(hrm * hr[ind])  +(ym * y[ind])  + m0));
w[ind] ~ poisson(exp((ew * e[ind])  +(mw * m[ind])  +(ow * o[ind])  +(rw * r[ind])  +(genderw * gender[ind])  +(hrw * hr[ind])  +(yw * y[ind])  + w0));
race[ind] ~ poisson(exp((agerace * age[ind])  +(wrace * w[ind])  +(erace * e[ind])  +(mrace * m[ind])  +(orace * o[ind])  +(rrace * r[ind])  +(genderrace * gender[ind])  +(hrrace * hr[ind])  +(yrace * y[ind])  + race0));
n[ind] ~ poisson(exp((agen * age[ind])  +(wn * w[ind])  +(en * e[ind])  +(mn * m[ind])  +(on * o[ind])  +(racen * race[ind])  +(gendern * gender[ind])  +(hrn * hr[ind])  +(yn * y[ind])  + n0));
}

}

