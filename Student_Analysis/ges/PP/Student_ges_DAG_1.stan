data{
int<lower = 0> N;
array[N] int<lower=0, upper=1>  sex;
array[N] int<lower=0, upper=1>  school;
array[N] int<lower=15, upper=22>  age;
array[N] int<lower=0, upper=4>  Medu;
array[N] int<lower=0, upper=4>  Fedu;
array[N] int<lower=1, upper=4>  reason;
array[N] int<lower=1, upper=3>  guardian;
array[N] int<lower=1, upper=4>  studytime;
array[N] int<lower=1, upper=4>  failures;
array[N] int<lower=0, upper=1>  higher;
array[N] int<lower=0, upper=1>  romantic;
array[N] int<lower=1, upper=5>  goout;
array[N] int<lower=1, upper=5>  Dalc;
array[N] int<lower=0, upper=75>  absences;
array[N] int<lower=0, upper=19>  G1;
array[N] int<lower=0, upper=19>  G2;
array[N] int<lower=0, upper=1>  G3;
}

transformed data {
}

parameters {
real sexstudytime;
real sexhigher;
real sex0;

real schoolage;
real schoolstudytime;
real schoolabsences;
real school0;

real ageguardian;
real agehigher;
real ageromantic;
real ageabsences;
real age0;

real Medusex;
real Meduschool;
real MeduFedu;
real Medureason;
real Meduhigher;
real MeduG1;

real Fedufailures;
real Fedu0;

real reason0;

real guardian0;

real studytimereason;
real studytime0;

real failuresage;
real failuresguardian;
real failureshigher;
real failures0;

real higherstudytime;
real higher0;

real romanticsex;
real romanticabsences;
real romantic0;

real gooutage;
real goout0;

real Dalcsex;
real Dalcgoout;
real Dalcabsences;
real DalcG1;

real absencesreason;
real absencesguardian;
real absences0;

real G1school;
real G1studytime;
real G1higher;
real G1G2;
real G1G3;
real G10;

real G2reason;
real G2failures;
real G2G3;
real G20;

real G3failures;
real G3goout;
real G3absences;
real G30;

}

transformed parameters {
}

model {
sexstudytime        ~ normal(0, 1);
sexhigher        ~ normal(0, 1);
sex0        ~ normal(0, 1);

schoolage        ~ normal(0, 1);
schoolstudytime        ~ normal(0, 1);
schoolabsences        ~ normal(0, 1);
school0        ~ normal(0, 1);

ageguardian        ~ normal(0, 1);
agehigher        ~ normal(0, 1);
ageromantic        ~ normal(0, 1);
ageabsences        ~ normal(0, 1);
age0        ~ normal(0, 1);

Medusex        ~ normal(0, 1);
Meduschool        ~ normal(0, 1);
MeduFedu        ~ normal(0, 1);
Medureason        ~ normal(0, 1);
Meduhigher        ~ normal(0, 1);
MeduG1        ~ normal(0, 1);

Fedufailures        ~ normal(0, 1);
Fedu0        ~ normal(0, 1);

reason0        ~ normal(0, 1);

guardian0        ~ normal(0, 1);

studytimereason        ~ normal(0, 1);
studytime0        ~ normal(0, 1);

failuresage        ~ normal(0, 1);
failuresguardian        ~ normal(0, 1);
failureshigher        ~ normal(0, 1);
failures0        ~ normal(0, 1);

higherstudytime        ~ normal(0, 1);
higher0        ~ normal(0, 1);

romanticsex        ~ normal(0, 1);
romanticabsences        ~ normal(0, 1);
romantic0        ~ normal(0, 1);

gooutage        ~ normal(0, 1);
goout0        ~ normal(0, 1);

Dalcsex        ~ normal(0, 1);
Dalcgoout        ~ normal(0, 1);
Dalcabsences        ~ normal(0, 1);
DalcG1        ~ normal(0, 1);

absencesreason        ~ normal(0, 1);
absencesguardian        ~ normal(0, 1);
absences0        ~ normal(0, 1);

G1school        ~ normal(0, 1);
G1studytime        ~ normal(0, 1);
G1higher        ~ normal(0, 1);
G1G2        ~ normal(0, 1);
G1G3        ~ normal(0, 1);
G10        ~ normal(0, 1);

G2reason        ~ normal(0, 1);
G2failures        ~ normal(0, 1);
G2G3        ~ normal(0, 1);
G20        ~ normal(0, 1);

G3failures        ~ normal(0, 1);
G3goout        ~ normal(0, 1);
G3absences        ~ normal(0, 1);
G30        ~ normal(0, 1);

for(ind in 1:N){Fedu[ind] ~ poisson(exp((MeduFedu * Medu[ind])  + Fedu0));
G1[ind] ~ poisson(exp((MeduG1 * Medu[ind])  +(DalcG1 * Dalc[ind])  + G10));
G2[ind] ~ poisson(exp((G1G2 * G1[ind])  + G20));
G3[ind] ~ bernoulli_logit((G1G3 * G1[ind])  +(G2G3 * G2[ind])  + G30);
school[ind] ~ bernoulli_logit((Meduschool * Medu[ind])  +(G1school * G1[ind])  + school0);
failures[ind] ~ poisson(exp((Fedufailures * Fedu[ind])  +(G2failures * G2[ind])  +(G3failures * G3[ind])  + failures0));
goout[ind] ~ poisson(exp((Dalcgoout * Dalc[ind])  +(G3goout * G3[ind])  + goout0));
age[ind] ~ poisson(exp((schoolage * school[ind])  +(failuresage * failures[ind])  +(gooutage * goout[ind])  + age0));
romantic[ind] ~ bernoulli_logit((ageromantic * age[ind])  + romantic0);
absences[ind] ~ poisson(exp((schoolabsences * school[ind])  +(ageabsences * age[ind])  +(romanticabsences * romantic[ind])  +(Dalcabsences * Dalc[ind])  +(G3absences * G3[ind])  + absences0));
sex[ind] ~ bernoulli_logit((Medusex * Medu[ind])  +(romanticsex * romantic[ind])  +(Dalcsex * Dalc[ind])  + sex0);
guardian[ind] ~ poisson(exp((ageguardian * age[ind])  +(failuresguardian * failures[ind])  +(absencesguardian * absences[ind])  + guardian0));
higher[ind] ~ bernoulli_logit((sexhigher * sex[ind])  +(agehigher * age[ind])  +(Meduhigher * Medu[ind])  +(failureshigher * failures[ind])  +(G1higher * G1[ind])  + higher0);
studytime[ind] ~ poisson(exp((sexstudytime * sex[ind])  +(schoolstudytime * school[ind])  +(higherstudytime * higher[ind])  +(G1studytime * G1[ind])  + studytime0));
reason[ind] ~ poisson(exp((Medureason * Medu[ind])  +(studytimereason * studytime[ind])  +(absencesreason * absences[ind])  +(G2reason * G2[ind])  + reason0));
}

}

