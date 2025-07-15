data{
int<lower = 0> N;
array[N] int<lower=1, upper=9>  age;
array[N] int<lower=0, upper=11>  b;
array[N] int<lower=0, upper=2>  c;
array[N] int<lower=0, upper=3>  d;
array[N] int<lower=0, upper=1>  e;
array[N] int<lower=0, upper=199>  f;
array[N] int<lower=0, upper=1>  g;
array[N] int<lower=0, upper=1>  h;
array[N] int<lower=0, upper=2>  i;
array[N] int<lower=1, upper=31>  j;
array[N] int<lower=0, upper=11>  k;
array[N] int<lower=0, upper=99>  l;
array[N] int<lower=1, upper=63>  m;
array[N] int<lower=0, upper=1>  n;
array[N] int<lower=0, upper=1>  o;
array[N] int<lower=0, upper=3>  p;
array[N] int<lower=0, upper=1>  label;
}

transformed data {
}

parameters {
real ageb;
real aged;
real agef;
real agei;
real agek;
real age0;

real b0;

real cage;
real cd;
real ck;
real c0;

real db;
real d0;

real e0;

real fd;
real fe;
real f0;

real gage;
real gb;
real gd;
real gf;
real gh;
real gi;
real gj;
real gk;
real gn;
real go;
real glabel;
real g0;

real hage;
real hb;
real hc;
real hd;
real he;
real hf;
real hi;
real hk;
real h0;

real id;
real i0;

real ji;
real jk;
real j0;

real kd;
real ke;
real kf;
real ki;
real k0;

real ld;
real lf;
real lj;
real lm;
real lo;
real llabel;

real mc;
real md;
real mf;
real mg;
real mj;
real mk;
real mp;
real mlabel;
real m0;

real nc;
real ne;
real nf;
real ni;
real nj;
real nk;
real nlabel;
real n0;

real on;
real o0;

real page;
real pb;
real pc;
real pg;
real ph;
real pi;
real pk;
real po;
real plabel;
real p0;

real labelage;
real labelc;
real labeld;
real labelf;
real labelh;
real labeli;
real label0;

}

transformed parameters {
}

model {
ageb        ~ normal(0, 1);
aged        ~ normal(0, 1);
agef        ~ normal(0, 1);
agei        ~ normal(0, 1);
agek        ~ normal(0, 1);
age0        ~ normal(0, 1);

b0        ~ normal(0, 1);

cage        ~ normal(0, 1);
cd        ~ normal(0, 1);
ck        ~ normal(0, 1);
c0        ~ normal(0, 1);

db        ~ normal(0, 1);
d0        ~ normal(0, 1);

e0        ~ normal(0, 1);

fd        ~ normal(0, 1);
fe        ~ normal(0, 1);
f0        ~ normal(0, 1);

gage        ~ normal(0, 1);
gb        ~ normal(0, 1);
gd        ~ normal(0, 1);
gf        ~ normal(0, 1);
gh        ~ normal(0, 1);
gi        ~ normal(0, 1);
gj        ~ normal(0, 1);
gk        ~ normal(0, 1);
gn        ~ normal(0, 1);
go        ~ normal(0, 1);
glabel        ~ normal(0, 1);
g0        ~ normal(0, 1);

hage        ~ normal(0, 1);
hb        ~ normal(0, 1);
hc        ~ normal(0, 1);
hd        ~ normal(0, 1);
he        ~ normal(0, 1);
hf        ~ normal(0, 1);
hi        ~ normal(0, 1);
hk        ~ normal(0, 1);
h0        ~ normal(0, 1);

id        ~ normal(0, 1);
i0        ~ normal(0, 1);

ji        ~ normal(0, 1);
jk        ~ normal(0, 1);
j0        ~ normal(0, 1);

kd        ~ normal(0, 1);
ke        ~ normal(0, 1);
kf        ~ normal(0, 1);
ki        ~ normal(0, 1);
k0        ~ normal(0, 1);

ld        ~ normal(0, 1);
lf        ~ normal(0, 1);
lj        ~ normal(0, 1);
lm        ~ normal(0, 1);
lo        ~ normal(0, 1);
llabel        ~ normal(0, 1);

mc        ~ normal(0, 1);
md        ~ normal(0, 1);
mf        ~ normal(0, 1);
mg        ~ normal(0, 1);
mj        ~ normal(0, 1);
mk        ~ normal(0, 1);
mp        ~ normal(0, 1);
mlabel        ~ normal(0, 1);
m0        ~ normal(0, 1);

nc        ~ normal(0, 1);
ne        ~ normal(0, 1);
nf        ~ normal(0, 1);
ni        ~ normal(0, 1);
nj        ~ normal(0, 1);
nk        ~ normal(0, 1);
nlabel        ~ normal(0, 1);
n0        ~ normal(0, 1);

on        ~ normal(0, 1);
o0        ~ normal(0, 1);

page        ~ normal(0, 1);
pb        ~ normal(0, 1);
pc        ~ normal(0, 1);
pg        ~ normal(0, 1);
ph        ~ normal(0, 1);
pi        ~ normal(0, 1);
pk        ~ normal(0, 1);
po        ~ normal(0, 1);
plabel        ~ normal(0, 1);
p0        ~ normal(0, 1);

labelage        ~ normal(0, 1);
labelc        ~ normal(0, 1);
labeld        ~ normal(0, 1);
labelf        ~ normal(0, 1);
labelh        ~ normal(0, 1);
labeli        ~ normal(0, 1);
label0        ~ normal(0, 1);

for(ind in 1:N){m[ind] ~ poisson(exp((lm * l[ind])  + m0));
p[ind] ~ poisson(exp((mp * m[ind])  + p0));
g[ind] ~ bernoulli_logit((mg * m[ind])  +(pg * p[ind])  + g0);
o[ind] ~ bernoulli_logit((go * g[ind])  +(lo * l[ind])  +(po * p[ind])  + o0);
n[ind] ~ bernoulli_logit((gn * g[ind])  +(on * o[ind])  + n0);
label[ind] ~ bernoulli_logit((glabel * g[ind])  +(llabel * l[ind])  +(mlabel * m[ind])  +(nlabel * n[ind])  +(plabel * p[ind])  + label0);
h[ind] ~ bernoulli_logit((gh * g[ind])  +(ph * p[ind])  +(labelh * label[ind])  + h0);
j[ind] ~ poisson(exp((gj * g[ind])  +(lj * l[ind])  +(mj * m[ind])  +(nj * n[ind])  + j0));
c[ind] ~ poisson(exp((hc * h[ind])  +(mc * m[ind])  +(nc * n[ind])  +(pc * p[ind])  +(labelc * label[ind])  + c0));
age[ind] ~ poisson(exp((cage * c[ind])  +(gage * g[ind])  +(hage * h[ind])  +(page * p[ind])  +(labelage * label[ind])  + age0));
k[ind] ~ poisson(exp((agek * age[ind])  +(ck * c[ind])  +(gk * g[ind])  +(hk * h[ind])  +(jk * j[ind])  +(mk * m[ind])  +(nk * n[ind])  +(pk * p[ind])  + k0));
f[ind] ~ poisson(exp((agef * age[ind])  +(gf * g[ind])  +(hf * h[ind])  +(kf * k[ind])  +(lf * l[ind])  +(mf * m[ind])  +(nf * n[ind])  +(labelf * label[ind])  + f0));
i[ind] ~ poisson(exp((agei * age[ind])  +(gi * g[ind])  +(hi * h[ind])  +(ji * j[ind])  +(ki * k[ind])  +(ni * n[ind])  +(pi * p[ind])  +(labeli * label[ind])  + i0));
d[ind] ~ poisson(exp((aged * age[ind])  +(cd * c[ind])  +(fd * f[ind])  +(gd * g[ind])  +(hd * h[ind])  +(id * i[ind])  +(kd * k[ind])  +(ld * l[ind])  +(md * m[ind])  +(labeld * label[ind])  + d0));
e[ind] ~ bernoulli_logit((fe * f[ind])  +(he * h[ind])  +(ke * k[ind])  +(ne * n[ind])  + e0);
b[ind] ~ poisson(exp((ageb * age[ind])  +(db * d[ind])  +(gb * g[ind])  +(hb * h[ind])  +(pb * p[ind])  + b0));
}

}

