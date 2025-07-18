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
real sexromantic;
real sexDalc;

real schoolage;
real schoolstudytime;
real schoolabsences;
real school0;

real agehigher;
real ageromantic;
real age0;

real Meduschool;
real Medureason;
real Meduhigher;
real Medu0;

real FeduMedu;

real reasonstudytime;
real reason0;

real guardianage;
real guardianfailures;
real guardianabsences;

real studytimehigher;
real studytimeG1;
real studytime0;

real failuresage;
real failureshigher;
real failuresG3;
real failures0;

real higher0;

real romantic0;

real gooutDalc;

real Dalcabsences;
real Dalc0;

real absencesage;
real absences0;

real G1higher;
real G1G2;
real G1G3;
real G10;

real G20;

real G3G2;
real G30;

}

transformed parameters {
}

model {
sexstudytime        ~ normal(0, 1);
sexromantic        ~ normal(0, 1);
sexDalc        ~ normal(0, 1);

schoolage        ~ normal(0, 1);
schoolstudytime        ~ normal(0, 1);
schoolabsences        ~ normal(0, 1);
school0        ~ normal(0, 1);

agehigher        ~ normal(0, 1);
ageromantic        ~ normal(0, 1);
age0        ~ normal(0, 1);

Meduschool        ~ normal(0, 1);
Medureason        ~ normal(0, 1);
Meduhigher        ~ normal(0, 1);
Medu0        ~ normal(0, 1);

FeduMedu        ~ normal(0, 1);

reasonstudytime        ~ normal(0, 1);
reason0        ~ normal(0, 1);

guardianage        ~ normal(0, 1);
guardianfailures        ~ normal(0, 1);
guardianabsences        ~ normal(0, 1);

studytimehigher        ~ normal(0, 1);
studytimeG1        ~ normal(0, 1);
studytime0        ~ normal(0, 1);

failuresage        ~ normal(0, 1);
failureshigher        ~ normal(0, 1);
failuresG3        ~ normal(0, 1);
failures0        ~ normal(0, 1);

higher0        ~ normal(0, 1);

romantic0        ~ normal(0, 1);

gooutDalc        ~ normal(0, 1);

Dalcabsences        ~ normal(0, 1);
Dalc0        ~ normal(0, 1);

absencesage        ~ normal(0, 1);
absences0        ~ normal(0, 1);

G1higher        ~ normal(0, 1);
G1G2        ~ normal(0, 1);
G1G3        ~ normal(0, 1);
G10        ~ normal(0, 1);

G20        ~ normal(0, 1);

G3G2        ~ normal(0, 1);
G30        ~ normal(0, 1);

for(ind in 1:N){Medu[ind] ~ poisson(exp((FeduMedu * Fedu[ind])  + Medu0));
reason[ind] ~ poisson(exp((Medureason * Medu[ind])  + reason0));
failures[ind] ~ poisson(exp((guardianfailures * guardian[ind])  + failures0));
Dalc[ind] ~ poisson(exp((sexDalc * sex[ind])  +(gooutDalc * goout[ind])  + Dalc0));
school[ind] ~ bernoulli_logit((Meduschool * Medu[ind])  + school0);
studytime[ind] ~ poisson(exp((sexstudytime * sex[ind])  +(schoolstudytime * school[ind])  +(reasonstudytime * reason[ind])  + studytime0));
absences[ind] ~ poisson(exp((schoolabsences * school[ind])  +(guardianabsences * guardian[ind])  +(Dalcabsences * Dalc[ind])  + absences0));
G1[ind] ~ poisson(exp((studytimeG1 * studytime[ind])  + G10));
G3[ind] ~ bernoulli_logit((failuresG3 * failures[ind])  +(G1G3 * G1[ind])  + G30);
age[ind] ~ poisson(exp((schoolage * school[ind])  +(guardianage * guardian[ind])  +(failuresage * failures[ind])  +(absencesage * absences[ind])  + age0));
higher[ind] ~ bernoulli_logit((agehigher * age[ind])  +(Meduhigher * Medu[ind])  +(studytimehigher * studytime[ind])  +(failureshigher * failures[ind])  +(G1higher * G1[ind])  + higher0);
romantic[ind] ~ bernoulli_logit((sexromantic * sex[ind])  +(ageromantic * age[ind])  + romantic0);
G2[ind] ~ poisson(exp((G1G2 * G1[ind])  +(G3G2 * G3[ind])  + G20));
}

}

