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
real sexMedu;
real sexstudytime;
real sexDalc;
real sex0;

real schoolage;
real schoolMedu;
real schoolstudytime;
real schoolabsences;
real schoolG1;

real agefailures;
real agehigher;
real ageromantic;
real agegoout;
real ageDalc;
real ageabsences;
real age0;

real MeduFedu;
real Medureason;
real Medu0;

real Fedu0;

real reasonabsences;
real reason0;

real guardianage;
real guardianFedu;
real guardianfailures;
real guardianabsences;

real studytimereason;
real studytimefailures;
real studytimehigher;
real studytimeabsences;
real studytimeG1;
real studytime0;

real failuresMedu;
real failureshigher;
real failuresG1;
real failuresG3;
real failures0;

real higherMedu;
real higher0;

real romanticsex;
real romanticabsences;
real romanticG2;
real romantic0;

real goout0;

real Dalcgoout;
real Dalc0;

real absencesDalc;
real absences0;

real G1higher;
real G1Dalc;
real G1G2;
real G1G3;
real G10;

real G2Medu;
real G2reason;
real G20;

real G3Medu;
real G3absences;
real G3G2;
real G30;

}

transformed parameters {
}

model {
sexMedu        ~ normal(0, 1);
sexstudytime        ~ normal(0, 1);
sexDalc        ~ normal(0, 1);
sex0        ~ normal(0, 1);

schoolage        ~ normal(0, 1);
schoolMedu        ~ normal(0, 1);
schoolstudytime        ~ normal(0, 1);
schoolabsences        ~ normal(0, 1);
schoolG1        ~ normal(0, 1);

agefailures        ~ normal(0, 1);
agehigher        ~ normal(0, 1);
ageromantic        ~ normal(0, 1);
agegoout        ~ normal(0, 1);
ageDalc        ~ normal(0, 1);
ageabsences        ~ normal(0, 1);
age0        ~ normal(0, 1);

MeduFedu        ~ normal(0, 1);
Medureason        ~ normal(0, 1);
Medu0        ~ normal(0, 1);

Fedu0        ~ normal(0, 1);

reasonabsences        ~ normal(0, 1);
reason0        ~ normal(0, 1);

guardianage        ~ normal(0, 1);
guardianFedu        ~ normal(0, 1);
guardianfailures        ~ normal(0, 1);
guardianabsences        ~ normal(0, 1);

studytimereason        ~ normal(0, 1);
studytimefailures        ~ normal(0, 1);
studytimehigher        ~ normal(0, 1);
studytimeabsences        ~ normal(0, 1);
studytimeG1        ~ normal(0, 1);
studytime0        ~ normal(0, 1);

failuresMedu        ~ normal(0, 1);
failureshigher        ~ normal(0, 1);
failuresG1        ~ normal(0, 1);
failuresG3        ~ normal(0, 1);
failures0        ~ normal(0, 1);

higherMedu        ~ normal(0, 1);
higher0        ~ normal(0, 1);

romanticsex        ~ normal(0, 1);
romanticabsences        ~ normal(0, 1);
romanticG2        ~ normal(0, 1);
romantic0        ~ normal(0, 1);

goout0        ~ normal(0, 1);

Dalcgoout        ~ normal(0, 1);
Dalc0        ~ normal(0, 1);

absencesDalc        ~ normal(0, 1);
absences0        ~ normal(0, 1);

G1higher        ~ normal(0, 1);
G1Dalc        ~ normal(0, 1);
G1G2        ~ normal(0, 1);
G1G3        ~ normal(0, 1);
G10        ~ normal(0, 1);

G2Medu        ~ normal(0, 1);
G2reason        ~ normal(0, 1);
G20        ~ normal(0, 1);

G3Medu        ~ normal(0, 1);
G3absences        ~ normal(0, 1);
G3G2        ~ normal(0, 1);
G30        ~ normal(0, 1);

for(ind in 1:N){age[ind] ~ poisson(exp((schoolage * school[ind])  +(guardianage * guardian[ind])  + age0));
romantic[ind] ~ bernoulli_logit((ageromantic * age[ind])  + romantic0);
sex[ind] ~ bernoulli_logit((romanticsex * romantic[ind])  + sex0);
studytime[ind] ~ poisson(exp((sexstudytime * sex[ind])  +(schoolstudytime * school[ind])  + studytime0));
failures[ind] ~ poisson(exp((agefailures * age[ind])  +(guardianfailures * guardian[ind])  +(studytimefailures * studytime[ind])  + failures0));
G1[ind] ~ poisson(exp((schoolG1 * school[ind])  +(studytimeG1 * studytime[ind])  +(failuresG1 * failures[ind])  + G10));
G3[ind] ~ bernoulli_logit((failuresG3 * failures[ind])  +(G1G3 * G1[ind])  + G30);
higher[ind] ~ bernoulli_logit((agehigher * age[ind])  +(studytimehigher * studytime[ind])  +(failureshigher * failures[ind])  +(G1higher * G1[ind])  + higher0);
G2[ind] ~ poisson(exp((romanticG2 * romantic[ind])  +(G1G2 * G1[ind])  +(G3G2 * G3[ind])  + G20));
Medu[ind] ~ poisson(exp((sexMedu * sex[ind])  +(schoolMedu * school[ind])  +(failuresMedu * failures[ind])  +(higherMedu * higher[ind])  +(G2Medu * G2[ind])  +(G3Medu * G3[ind])  + Medu0));
Fedu[ind] ~ poisson(exp((MeduFedu * Medu[ind])  +(guardianFedu * guardian[ind])  + Fedu0));
reason[ind] ~ poisson(exp((Medureason * Medu[ind])  +(studytimereason * studytime[ind])  +(G2reason * G2[ind])  + reason0));
absences[ind] ~ poisson(exp((schoolabsences * school[ind])  +(ageabsences * age[ind])  +(reasonabsences * reason[ind])  +(guardianabsences * guardian[ind])  +(studytimeabsences * studytime[ind])  +(romanticabsences * romantic[ind])  +(G3absences * G3[ind])  + absences0));
Dalc[ind] ~ poisson(exp((sexDalc * sex[ind])  +(ageDalc * age[ind])  +(absencesDalc * absences[ind])  +(G1Dalc * G1[ind])  + Dalc0));
goout[ind] ~ poisson(exp((agegoout * age[ind])  +(Dalcgoout * Dalc[ind])  + goout0));
}

}

