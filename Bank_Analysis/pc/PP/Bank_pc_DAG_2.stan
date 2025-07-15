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
real agec;
real agef;
real ageg;
real agek;


real bd;
real bg;
real bh;
real bl;
real b0;


real cd;
real ch;
real ci;
real ck;
real cm;
real clabel;
real c0;


real df;
real dg;
real dh;
real di;
real dk;
real dlabel;
real d0;


real ef;
real eh;


real fg;
real fh;
real fk;
real flabel;
real f0;


real gh;
real gi;
real gk;
real gn;
real go;
real gp;
real glabel;
real g0;


real hp;
real hlabel;
real h0;


real ik;
real il;
real ilabel;
real i0;


real jk;
real jm;
real jn;
real jo;


real kn;
real ko;
real kp;
real k0;


real lm;
real llabel;
real l0;


real mk;
real mp;
real m0;


real np;
real nlabel;
real n0;


real on;
real op;
real olabel;
real o0;


real plabel;
real p0;


real label0;


}

transformed parameters {

}

model {

ageb        ~ normal(0, 1);
agec        ~ normal(0, 1);
agef        ~ normal(0, 1);
ageg        ~ normal(0, 1);
agek        ~ normal(0, 1);


bd        ~ normal(0, 1);
bg        ~ normal(0, 1);
bh        ~ normal(0, 1);
bl        ~ normal(0, 1);
b0        ~ normal(0, 1);


cd        ~ normal(0, 1);
ch        ~ normal(0, 1);
ci        ~ normal(0, 1);
ck        ~ normal(0, 1);
cm        ~ normal(0, 1);
clabel        ~ normal(0, 1);
c0        ~ normal(0, 1);


df        ~ normal(0, 1);
dg        ~ normal(0, 1);
dh        ~ normal(0, 1);
di        ~ normal(0, 1);
dk        ~ normal(0, 1);
dlabel        ~ normal(0, 1);
d0        ~ normal(0, 1);


ef        ~ normal(0, 1);
eh        ~ normal(0, 1);


fg        ~ normal(0, 1);
fh        ~ normal(0, 1);
fk        ~ normal(0, 1);
flabel        ~ normal(0, 1);
f0        ~ normal(0, 1);


gh        ~ normal(0, 1);
gi        ~ normal(0, 1);
gk        ~ normal(0, 1);
gn        ~ normal(0, 1);
go        ~ normal(0, 1);
gp        ~ normal(0, 1);
glabel        ~ normal(0, 1);
g0        ~ normal(0, 1);


hp        ~ normal(0, 1);
hlabel        ~ normal(0, 1);
h0        ~ normal(0, 1);


ik        ~ normal(0, 1);
il        ~ normal(0, 1);
ilabel        ~ normal(0, 1);
i0        ~ normal(0, 1);


jk        ~ normal(0, 1);
jm        ~ normal(0, 1);
jn        ~ normal(0, 1);
jo        ~ normal(0, 1);


kn        ~ normal(0, 1);
ko        ~ normal(0, 1);
kp        ~ normal(0, 1);
k0        ~ normal(0, 1);


lm        ~ normal(0, 1);
llabel        ~ normal(0, 1);
l0        ~ normal(0, 1);


mk        ~ normal(0, 1);
mp        ~ normal(0, 1);
m0        ~ normal(0, 1);


np        ~ normal(0, 1);
nlabel        ~ normal(0, 1);
n0        ~ normal(0, 1);


on        ~ normal(0, 1);
op        ~ normal(0, 1);
olabel        ~ normal(0, 1);
o0        ~ normal(0, 1);


plabel        ~ normal(0, 1);
p0        ~ normal(0, 1);


label0        ~ normal(0, 1);


for(ind in 1:N){
b[ind] ~ poisson(exp((ageb * age[ind])  + b0));
c[ind] ~ poisson(exp((agec * age[ind])  + c0));
d[ind] ~ poisson(exp((bd * b[ind])  +(cd * c[ind])  + d0));
f[ind] ~ poisson(exp((agef * age[ind])  +(df * d[ind])  +(ef * e[ind])  + f0));
g[ind] ~ bernoulli_logit((ageg * age[ind])  +(bg * b[ind])  +(dg * d[ind])  +(fg * f[ind])  + g0);
h[ind] ~ bernoulli_logit((bh * b[ind])  +(ch * c[ind])  +(dh * d[ind])  +(eh * e[ind])  +(fh * f[ind])  +(gh * g[ind])  + h0);
i[ind] ~ poisson(exp((ci * c[ind])  +(di * d[ind])  +(gi * g[ind])  + i0));
l[ind] ~ poisson(exp((bl * b[ind])  +(il * i[ind])  + l0));
m[ind] ~ poisson(exp((cm * c[ind])  +(jm * j[ind])  +(lm * l[ind])  + m0));
k[ind] ~ poisson(exp((agek * age[ind])  +(ck * c[ind])  +(dk * d[ind])  +(fk * f[ind])  +(gk * g[ind])  +(ik * i[ind])  +(jk * j[ind])  +(mk * m[ind])  + k0));
o[ind] ~ bernoulli_logit((go * g[ind])  +(jo * j[ind])  +(ko * k[ind])  + o0);
n[ind] ~ bernoulli_logit((gn * g[ind])  +(jn * j[ind])  +(kn * k[ind])  +(on * o[ind])  + n0);
p[ind] ~ poisson(exp((gp * g[ind])  +(hp * h[ind])  +(kp * k[ind])  +(mp * m[ind])  +(np * n[ind])  +(op * o[ind])  + p0));
label[ind] ~ bernoulli_logit((clabel * c[ind])  +(dlabel * d[ind])  +(flabel * f[ind])  +(glabel * g[ind])  +(hlabel * h[ind])  +(ilabel * i[ind])  +(llabel * l[ind])  +(nlabel * n[ind])  +(olabel * o[ind])  +(plabel * p[ind])  + label0);
}

}
