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
real agei;
real age0;

real b0;

real cage;
real cd;
real ch;
real co;
real c0;

real db;
real d0;

real eh;
real e0;

real fage;
real fc;
real fd;
real fe;
real fh;
real flabel;
real f0;

real gage;
real gb;
real gc;
real gd;
real gf;
real gh;
real gi;
real gk;
real go;
real gp;
real glabel;

real hage;
real hb;
real hd;
real hi;
real h0;

real id;
real i0;

real jage;
real ji;
real jk;
real jl;
real jo;
real jp;
real j0;

real kage;
real kc;
real kd;
real ke;
real kf;
real kh;
real ki;
real km;
real ko;
real kp;
real k0;

real ld;
real lf;
real lo;
real llabel;
real l0;

real mc;
real md;
real mf;
real mj;
real ml;
real m0;

real ne;
real ni;
real n0;

real on;
real o0;

real page;
real pb;
real pf;
real ph;
real pi;
real pm;
real po;
real plabel;
real p0;

real labelc;
real labeld;
real labelh;
real labeli;
real labelo;
real label0;

}

transformed parameters {
}

model {
ageb        ~ normal(0, 1);
aged        ~ normal(0, 1);
agei        ~ normal(0, 1);
age0        ~ normal(0, 1);

b0        ~ normal(0, 1);

cage        ~ normal(0, 1);
cd        ~ normal(0, 1);
ch        ~ normal(0, 1);
co        ~ normal(0, 1);
c0        ~ normal(0, 1);

db        ~ normal(0, 1);
d0        ~ normal(0, 1);

eh        ~ normal(0, 1);
e0        ~ normal(0, 1);

fage        ~ normal(0, 1);
fc        ~ normal(0, 1);
fd        ~ normal(0, 1);
fe        ~ normal(0, 1);
fh        ~ normal(0, 1);
flabel        ~ normal(0, 1);
f0        ~ normal(0, 1);

gage        ~ normal(0, 1);
gb        ~ normal(0, 1);
gc        ~ normal(0, 1);
gd        ~ normal(0, 1);
gf        ~ normal(0, 1);
gh        ~ normal(0, 1);
gi        ~ normal(0, 1);
gk        ~ normal(0, 1);
go        ~ normal(0, 1);
gp        ~ normal(0, 1);
glabel        ~ normal(0, 1);

hage        ~ normal(0, 1);
hb        ~ normal(0, 1);
hd        ~ normal(0, 1);
hi        ~ normal(0, 1);
h0        ~ normal(0, 1);

id        ~ normal(0, 1);
i0        ~ normal(0, 1);

jage        ~ normal(0, 1);
ji        ~ normal(0, 1);
jk        ~ normal(0, 1);
jl        ~ normal(0, 1);
jo        ~ normal(0, 1);
jp        ~ normal(0, 1);
j0        ~ normal(0, 1);

kage        ~ normal(0, 1);
kc        ~ normal(0, 1);
kd        ~ normal(0, 1);
ke        ~ normal(0, 1);
kf        ~ normal(0, 1);
kh        ~ normal(0, 1);
ki        ~ normal(0, 1);
km        ~ normal(0, 1);
ko        ~ normal(0, 1);
kp        ~ normal(0, 1);
k0        ~ normal(0, 1);

ld        ~ normal(0, 1);
lf        ~ normal(0, 1);
lo        ~ normal(0, 1);
llabel        ~ normal(0, 1);
l0        ~ normal(0, 1);

mc        ~ normal(0, 1);
md        ~ normal(0, 1);
mf        ~ normal(0, 1);
mj        ~ normal(0, 1);
ml        ~ normal(0, 1);
m0        ~ normal(0, 1);

ne        ~ normal(0, 1);
ni        ~ normal(0, 1);
n0        ~ normal(0, 1);

on        ~ normal(0, 1);
o0        ~ normal(0, 1);

page        ~ normal(0, 1);
pb        ~ normal(0, 1);
pf        ~ normal(0, 1);
ph        ~ normal(0, 1);
pi        ~ normal(0, 1);
pm        ~ normal(0, 1);
po        ~ normal(0, 1);
plabel        ~ normal(0, 1);
p0        ~ normal(0, 1);

labelc        ~ normal(0, 1);
labeld        ~ normal(0, 1);
labelh        ~ normal(0, 1);
labeli        ~ normal(0, 1);
labelo        ~ normal(0, 1);
label0        ~ normal(0, 1);

for(ind in 1:N){