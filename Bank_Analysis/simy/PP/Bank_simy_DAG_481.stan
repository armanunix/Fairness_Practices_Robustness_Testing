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
array[N] int<lower=0, upper=2>  i1;
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

real i1age;
real i1b;
real i1c;
real i1d;
real i1f;
real i1g;
real i1h;
real i10;

real jage;
real jb;
real jf;
real jg;
real ji1;
real j0;

real kage;
real kb;
real kc;
real kd;
real ke;
real kf;
real kg;
real kh;
real ki1;
real kj;
real k0;

real lage;
real lb;
real ld;
real lf;
real lg;
real lj;
real lk;
real l0;

real mage;
real mb;
real mc;
real mf;
real mg;
real mj;
real mk;
real ml;
real mlabel;
real m0;

real nage;
real nb;
real nf;
real ng;
real nj;
real nk;
real nl;
real no;

real oage;
real ob;
real oe;
real of;
real og;
real oi1;
real oj;
real ok;
real ol;
real op;
real olabel;
real o0;

real page;
real pb;
real pf;
real pg;
real ph;
real pi1;
real pj;
real pk;
real pl;
real pm;
real plabel;
real p0;

real labelage;
real labelb;
real labelc;
real labeld;
real labelf;
real labelg;
real labelh;
real labeli1;
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

i1age        ~ normal(0, 1);
i1b        ~ normal(0, 1);
i1c        ~ normal(0, 1);
i1d        ~ normal(0, 1);
i1f        ~ normal(0, 1);
i1g        ~ normal(0, 1);
i1h        ~ normal(0, 1);
i10        ~ normal(0, 1);

jage        ~ normal(0, 1);
jb        ~ normal(0, 1);
jf        ~ normal(0, 1);
jg        ~ normal(0, 1);
ji1        ~ normal(0, 1);
j0        ~ normal(0, 1);

kage        ~ normal(0, 1);
kb        ~ normal(0, 1);
kc        ~ normal(0, 1);
kd        ~ normal(0, 1);
ke        ~ normal(0, 1);
kf        ~ normal(0, 1);
kg        ~ normal(0, 1);
kh        ~ normal(0, 1);
ki1        ~ normal(0, 1);
kj        ~ normal(0, 1);
k0        ~ normal(0, 1);

lage        ~ normal(0, 1);
lb        ~ normal(0, 1);
ld        ~ normal(0, 1);
lf        ~ normal(0, 1);
lg        ~ normal(0, 1);
lj        ~ normal(0, 1);
lk        ~ normal(0, 1);
l0        ~ normal(0, 1);

mage        ~ normal(0, 1);
mb        ~ normal(0, 1);
mc        ~ normal(0, 1);
mf        ~ normal(0, 1);
mg        ~ normal(0, 1);
mj        ~ normal(0, 1);
mk        ~ normal(0, 1);
ml        ~ normal(0, 1);
mlabel        ~ normal(0, 1);
m0        ~ normal(0, 1);

nage        ~ normal(0, 1);
nb        ~ normal(0, 1);
nf        ~ normal(0, 1);
ng        ~ normal(0, 1);
nj        ~ normal(0, 1);
nk        ~ normal(0, 1);
nl        ~ normal(0, 1);
no        ~ normal(0, 1);

oage        ~ normal(0, 1);
ob        ~ normal(0, 1);
oe        ~ normal(0, 1);
of        ~ normal(0, 1);
og        ~ normal(0, 1);
oi1        ~ normal(0, 1);
oj        ~ normal(0, 1);
ok        ~ normal(0, 1);
ol        ~ normal(0, 1);
op        ~ normal(0, 1);
olabel        ~ normal(0, 1);
o0        ~ normal(0, 1);

page        ~ normal(0, 1);
pb        ~ normal(0, 1);
pf        ~ normal(0, 1);
pg        ~ normal(0, 1);
ph        ~ normal(0, 1);
pi1        ~ normal(0, 1);
pj        ~ normal(0, 1);
pk        ~ normal(0, 1);
pl        ~ normal(0, 1);
pm        ~ normal(0, 1);
plabel        ~ normal(0, 1);
p0        ~ normal(0, 1);

labelage        ~ normal(0, 1);
labelb        ~ normal(0, 1);
labelc        ~ normal(0, 1);
labeld        ~ normal(0, 1);
labelf        ~ normal(0, 1);
labelg        ~ normal(0, 1);
labelh        ~ normal(0, 1);
labeli1        ~ normal(0, 1);
labelj        ~ normal(0, 1);
labelk        ~ normal(0, 1);
labell        ~ normal(0, 1);
label0        ~ normal(0, 1);

for(ind in 1:N){o[ind] ~ bernoulli_logit((no * n[ind])  + o0);
p[ind] ~ poisson(exp((op * o[ind])  + p0));
m[ind] ~ poisson(exp((pm * p[ind])  + m0));
label[ind] ~ bernoulli_logit((mlabel * m[ind])  +(olabel * o[ind])  +(plabel * p[ind])  + label0);
l[ind] ~ poisson(exp((ml * m[ind])  +(nl * n[ind])  +(ol * o[ind])  +(pl * p[ind])  +(labell * label[ind])  + l0));
k[ind] ~ poisson(exp((lk * l[ind])  +(mk * m[ind])  +(nk * n[ind])  +(ok * o[ind])  +(pk * p[ind])  +(labelk * label[ind])  + k0));
j[ind] ~ poisson(exp((kj * k[ind])  +(lj * l[ind])  +(mj * m[ind])  +(nj * n[ind])  +(oj * o[ind])  +(pj * p[ind])  +(labelj * label[ind])  + j0));
i1[ind] ~ poisson(exp((ji1 * j[ind])  +(ki1 * k[ind])  +(oi1 * o[ind])  +(pi1 * p[ind])  +(labeli1 * label[ind])  + i10));
h[ind] ~ bernoulli_logit((i1h * i1[ind])  +(kh * k[ind])  +(ph * p[ind])  +(labelh * label[ind])  + h0);
f[ind] ~ poisson(exp((hf * h[ind])  +(i1f * i1[ind])  +(jf * j[ind])  +(kf * k[ind])  +(lf * l[ind])  +(mf * m[ind])  +(nf * n[ind])  +(of * o[ind])  +(pf * p[ind])  +(labelf * label[ind])  + f0));
g[ind] ~ bernoulli_logit((fg * f[ind])  +(hg * h[ind])  +(i1g * i1[ind])  +(jg * j[ind])  +(kg * k[ind])  +(lg * l[ind])  +(mg * m[ind])  +(ng * n[ind])  +(og * o[ind])  +(pg * p[ind])  +(labelg * label[ind])  + g0);
d[ind] ~ poisson(exp((fd * f[ind])  +(gd * g[ind])  +(hd * h[ind])  +(i1d * i1[ind])  +(kd * k[ind])  +(ld * l[ind])  +(labeld * label[ind])  + d0));
e[ind] ~ bernoulli_logit((fe * f[ind])  +(he * h[ind])  +(ke * k[ind])  +(oe * o[ind])  + e0);
c[ind] ~ poisson(exp((dc * d[ind])  +(fc * f[ind])  +(gc * g[ind])  +(hc * h[ind])  +(i1c * i1[ind])  +(kc * k[ind])  +(mc * m[ind])  +(labelc * label[ind])  + c0));
age[ind] ~ poisson(exp((cage * c[ind])  +(dage * d[ind])  +(eage * e[ind])  +(fage * f[ind])  +(gage * g[ind])  +(hage * h[ind])  +(i1age * i1[ind])  +(jage * j[ind])  +(kage * k[ind])  +(lage * l[ind])  +(mage * m[ind])  +(nage * n[ind])  +(oage * o[ind])  +(page * p[ind])  +(labelage * label[ind])  + age0));
b[ind] ~ poisson(exp((ageb * age[ind])  +(cb * c[ind])  +(db * d[ind])  +(eb * e[ind])  +(fb * f[ind])  +(gb * g[ind])  +(hb * h[ind])  +(i1b * i1[ind])  +(jb * j[ind])  +(kb * k[ind])  +(lb * l[ind])  +(mb * m[ind])  +(nb * n[ind])  +(ob * o[ind])  +(pb * p[ind])  +(labelb * label[ind])  + b0));
}

}

