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
real age0;

real b0;

real cage;
real cb;
real c0;

real dage;
real db;
real dc;
real d0;

real eage;
real eb;
real e0;

real fage;
real fb;
real fc;
real fd;
real fe;
real fg;
real f0;

real gage;
real gb;
real gc;
real gd;
real g0;

real hage;
real hb;
real hc;
real hd;
real he;
real hf;
real hg;
real h0;

real iage;
real ib;
real ic;
real id;
real if;
real ig;
real ih;
real i0;

real jage;
real jb;
real jf;
real jg;
real ji;
real jk;
real jl;
real j0;

real kage;
real kb;
real kc;
real kd;
real ke;
real kf;
real kg;
real kh;
real ki;
real kl;
real k0;

real lage;
real lb;
real ld;
real lf;
real lg;
real l0;

real mage;
real mb;
real mc;
real mf;
real mg;
real mj;
real mk;
real ml;
real mp;
real mlabel;

real nage;
real nb;
real nf;
real ng;
real nj;
real nk;
real nl;
real n0;

real oage;
real ob;
real oe;
real of;
real og;
real oi;
real oj;
real ok;
real ol;
real on;
real olabel;
real o0;

real page;
real pb;
real pf;
real pg;
real ph;
real pi;
real pj;
real pk;
real pl;
real po;
real plabel;
real p0;

real labelage;
real labelb;
real labelc;
real labeld;
real labelf;
real labelg;
real labelh;
real labeli;
real labelj;
real labelk;
real labell;
real label0;

}

transformed parameters {
}

model {
ageb        ~ normal(0, 1);
age0        ~ normal(0, 1);

b0        ~ normal(0, 1);

cage        ~ normal(0, 1);
cb        ~ normal(0, 1);
c0        ~ normal(0, 1);

dage        ~ normal(0, 1);
db        ~ normal(0, 1);
dc        ~ normal(0, 1);
d0        ~ normal(0, 1);

eage        ~ normal(0, 1);
eb        ~ normal(0, 1);
e0        ~ normal(0, 1);

fage        ~ normal(0, 1);
fb        ~ normal(0, 1);
fc        ~ normal(0, 1);
fd        ~ normal(0, 1);
fe        ~ normal(0, 1);
fg        ~ normal(0, 1);
f0        ~ normal(0, 1);

gage        ~ normal(0, 1);
gb        ~ normal(0, 1);
gc        ~ normal(0, 1);
gd        ~ normal(0, 1);
g0        ~ normal(0, 1);

hage        ~ normal(0, 1);
hb        ~ normal(0, 1);
hc        ~ normal(0, 1);
hd        ~ normal(0, 1);
he        ~ normal(0, 1);
hf        ~ normal(0, 1);
hg        ~ normal(0, 1);
h0        ~ normal(0, 1);

iage        ~ normal(0, 1);
ib        ~ normal(0, 1);
ic        ~ normal(0, 1);
id        ~ normal(0, 1);
if        ~ normal(0, 1);
ig        ~ normal(0, 1);
ih        ~ normal(0, 1);
i0        ~ normal(0, 1);

jage        ~ normal(0, 1);
jb        ~ normal(0, 1);
jf        ~ normal(0, 1);
jg        ~ normal(0, 1);
ji        ~ normal(0, 1);
jk        ~ normal(0, 1);
jl        ~ normal(0, 1);
j0        ~ normal(0, 1);

kage        ~ normal(0, 1);
kb        ~ normal(0, 1);
kc        ~ normal(0, 1);
kd        ~ normal(0, 1);
ke        ~ normal(0, 1);
kf        ~ normal(0, 1);
kg        ~ normal(0, 1);
kh        ~ normal(0, 1);
ki        ~ normal(0, 1);
kl        ~ normal(0, 1);
k0        ~ normal(0, 1);

lage        ~ normal(0, 1);
lb        ~ normal(0, 1);
ld        ~ normal(0, 1);
lf        ~ normal(0, 1);
lg        ~ normal(0, 1);
l0        ~ normal(0, 1);

mage        ~ normal(0, 1);
mb        ~ normal(0, 1);
mc        ~ normal(0, 1);
mf        ~ normal(0, 1);
mg        ~ normal(0, 1);
mj        ~ normal(0, 1);
mk        ~ normal(0, 1);
ml        ~ normal(0, 1);
mp        ~ normal(0, 1);
mlabel        ~ normal(0, 1);

nage        ~ normal(0, 1);
nb        ~ normal(0, 1);
nf        ~ normal(0, 1);
ng        ~ normal(0, 1);
nj        ~ normal(0, 1);
nk        ~ normal(0, 1);
nl        ~ normal(0, 1);
n0        ~ normal(0, 1);

oage        ~ normal(0, 1);
ob        ~ normal(0, 1);
oe        ~ normal(0, 1);
of        ~ normal(0, 1);
og        ~ normal(0, 1);
oi        ~ normal(0, 1);
oj        ~ normal(0, 1);
ok        ~ normal(0, 1);
ol        ~ normal(0, 1);
on        ~ normal(0, 1);
olabel        ~ normal(0, 1);
o0        ~ normal(0, 1);

page        ~ normal(0, 1);
pb        ~ normal(0, 1);
pf        ~ normal(0, 1);
pg        ~ normal(0, 1);
ph        ~ normal(0, 1);
pi        ~ normal(0, 1);
pj        ~ normal(0, 1);
pk        ~ normal(0, 1);
pl        ~ normal(0, 1);
po        ~ normal(0, 1);
plabel        ~ normal(0, 1);
p0        ~ normal(0, 1);

labelage        ~ normal(0, 1);
labelb        ~ normal(0, 1);
labelc        ~ normal(0, 1);
labeld        ~ normal(0, 1);
labelf        ~ normal(0, 1);
labelg        ~ normal(0, 1);
labelh        ~ normal(0, 1);
labeli        ~ normal(0, 1);
labelj        ~ normal(0, 1);
labelk        ~ normal(0, 1);
labell        ~ normal(0, 1);
label0        ~ normal(0, 1);

for(ind in 1:N){p[ind] ~ poisson(exp((mp * m[ind])  + p0));
o[ind] ~ bernoulli_logit((po * p[ind])  + o0);
label[ind] ~ bernoulli_logit((mlabel * m[ind])  +(olabel * o[ind])  +(plabel * p[ind])  + label0);
n[ind] ~ bernoulli_logit((on * o[ind])  + n0);
j[ind] ~ poisson(exp((mj * m[ind])  +(nj * n[ind])  +(oj * o[ind])  +(pj * p[ind])  +(labelj * label[ind])  + j0));
k[ind] ~ poisson(exp((jk * j[ind])  +(mk * m[ind])  +(nk * n[ind])  +(ok * o[ind])  +(pk * p[ind])  +(labelk * label[ind])  + k0));
l[ind] ~ poisson(exp((jl * j[ind])  +(kl * k[ind])  +(ml * m[ind])  +(nl * n[ind])  +(ol * o[ind])  +(pl * p[ind])  +(labell * label[ind])  + l0));
i[ind] ~ poisson(exp((ji * j[ind])  +(ki * k[ind])  +(oi * o[ind])  +(pi * p[ind])  +(labeli * label[ind])  + i0));
h[ind] ~ bernoulli_logit((ih * i[ind])  +(kh * k[ind])  +(ph * p[ind])  +(labelh * label[ind])  + h0);
f[ind] ~ poisson(exp((hf * h[ind])  +(if * i[ind])  +(jf * j[ind])  +(kf * k[ind])  +(lf * l[ind])  +(mf * m[ind])  +(nf * n[ind])  +(of * o[ind])  +(pf * p[ind])  +(labelf * label[ind])  + f0));
g[ind] ~ bernoulli_logit((fg * f[ind])  +(hg * h[ind])  +(ig * i[ind])  +(jg * j[ind])  +(kg * k[ind])  +(lg * l[ind])  +(mg * m[ind])  +(ng * n[ind])  +(og * o[ind])  +(pg * p[ind])  +(labelg * label[ind])  + g0);
d[ind] ~ poisson(exp((fd * f[ind])  +(gd * g[ind])  +(hd * h[ind])  +(id * i[ind])  +(kd * k[ind])  +(ld * l[ind])  +(labeld * label[ind])  + d0));
e[ind] ~ bernoulli_logit((fe * f[ind])  +(he * h[ind])  +(ke * k[ind])  +(oe * o[ind])  + e0);
c[ind] ~ poisson(exp((dc * d[ind])  +(fc * f[ind])  +(gc * g[ind])  +(hc * h[ind])  +(ic * i[ind])  +(kc * k[ind])  +(mc * m[ind])  +(labelc * label[ind])  + c0));
age[ind] ~ poisson(exp((cage * c[ind])  +(dage * d[ind])  +(eage * e[ind])  +(fage * f[ind])  +(gage * g[ind])  +(hage * h[ind])  +(iage * i[ind])  +(jage * j[ind])  +(kage * k[ind])  +(lage * l[ind])  +(mage * m[ind])  +(nage * n[ind])  +(oage * o[ind])  +(page * p[ind])  +(labelage * label[ind])  + age0));
b[ind] ~ poisson(exp((ageb * age[ind])  +(cb * c[ind])  +(db * d[ind])  +(eb * e[ind])  +(fb * f[ind])  +(gb * g[ind])  +(hb * h[ind])  +(ib * i[ind])  +(jb * j[ind])  +(kb * k[ind])  +(lb * l[ind])  +(mb * m[ind])  +(nb * n[ind])  +(ob * o[ind])  +(pb * p[ind])  +(labelb * label[ind])  + b0));
}

}

