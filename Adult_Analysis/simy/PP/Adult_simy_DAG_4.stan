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
real agey;
real age0;

real w0;

real er;
real ehr;
real ey;
real e0;

real mw;
real m0;

real oage;
real ow;
real oe;
real or;
real ohr;
real oy;
real o0;

real rage;
real rw;
real rm;
real rrace;
real ry;
real r0;

real racew;
real racem;
real race0;

real genderage;
real genderw;
real genderm;
real gendero;
real genderr;
real genderrace;
real genderhr;
real gendery;

real hrw;
real hrm;
real hrr;
real hry;
real hr0;

real ne;
real nm;
real nrace;

real ym;
real yrace;
real y0;

real<lower=0> sigma_h_Sq;
}

transformed parameters {
real<lower=0> sigma_h;
sigma_h = sqrt(sigma_h_Sq);
}

model {
agem        ~ normal(0, 1);
agey        ~ normal(0, 1);
age0        ~ normal(0, 1);

w0        ~ normal(0, 1);

er        ~ normal(0, 1);
ehr        ~ normal(0, 1);
ey        ~ normal(0, 1);
e0        ~ normal(0, 1);

mw        ~ normal(0, 1);
m0        ~ normal(0, 1);

oage        ~ normal(0, 1);
ow        ~ normal(0, 1);
oe        ~ normal(0, 1);
or        ~ normal(0, 1);
ohr        ~ normal(0, 1);
oy        ~ normal(0, 1);
o0        ~ normal(0, 1);

rage        ~ normal(0, 1);
rw        ~ normal(0, 1);
rm        ~ normal(0, 1);
rrace        ~ normal(0, 1);
ry        ~ normal(0, 1);
r0        ~ normal(0, 1);

racew        ~ normal(0, 1);
racem        ~ normal(0, 1);
race0        ~ normal(0, 1);

genderage        ~ normal(0, 1);
genderw        ~ normal(0, 1);
genderm        ~ normal(0, 1);
gendero        ~ normal(0, 1);
genderr        ~ normal(0, 1);
genderrace        ~ normal(0, 1);
genderhr        ~ normal(0, 1);
gendery        ~ normal(0, 1);

hrw        ~ normal(0, 1);
hrm        ~ normal(0, 1);
hrr        ~ normal(0, 1);
hry        ~ normal(0, 1);
hr0        ~ normal(0, 1);

ne        ~ normal(0, 1);
nm        ~ normal(0, 1);
nrace        ~ normal(0, 1);

ym        ~ normal(0, 1);
yrace        ~ normal(0, 1);
y0        ~ normal(0, 1);

sigma_h_Sq ~ inv_gamma(1, 1);

for(ind in 1:N){o[ind] ~ poisson(exp((gendero * gender[ind])  + o0));
e[ind] ~ poisson(exp((oe * o[ind])  +(ne * n[ind])  + e0));
hr[ind] ~ normal((ehr * e[ind])  +(ohr * o[ind])  +(genderhr * gender[ind])  + hr0, sigma_h);
r[ind] ~ poisson(exp((er * e[ind])  +(or * o[ind])  +(genderr * gender[ind])  +(hrr * hr[ind])  + r0));
age[ind] ~ poisson(exp((oage * o[ind])  +(rage * r[ind])  +(genderage * gender[ind])  + age0));
y[ind] ~ bernoulli_logit((agey * age[ind])  +(ey * e[ind])  +(oy * o[ind])  +(ry * r[ind])  +(gendery * gender[ind])  +(hry * hr[ind])  + y0);
race[ind] ~ poisson(exp((rrace * r[ind])  +(genderrace * gender[ind])  +(nrace * n[ind])  +(yrace * y[ind])  + race0));
m[ind] ~ poisson(exp((agem * age[ind])  +(rm * r[ind])  +(racem * race[ind])  +(genderm * gender[ind])  +(hrm * hr[ind])  +(nm * n[ind])  +(ym * y[ind])  + m0));
w[ind] ~ poisson(exp((mw * m[ind])  +(ow * o[ind])  +(rw * r[ind])  +(racew * race[ind])  +(genderw * gender[ind])  +(hrw * hr[ind])  + w0));
}

}

