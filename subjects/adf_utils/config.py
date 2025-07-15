# adapted from ADF https://github.com/pxzhang94/adf
class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain",
                                                                      "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    # note that for feature 9 (gender): 1,3,4 are male (replaced with 0) and 2.5 are female (replace with 1)
    params = 20

    input_bounds = []
    input_bounds.append([1, 4])
    input_bounds.append([4, 72])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([250, 18424])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([1, 3])
    input_bounds.append([1, 4])
    input_bounds.append([1, 4])
    input_bounds.append([19, 75])
    input_bounds.append([1, 3])
    input_bounds.append([1, 3])
    input_bounds.append([1, 4])
    input_bounds.append([1, 4])
    input_bounds.append([1, 2])
    input_bounds.append([1, 2])
    input_bounds.append([1, 2])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                    "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]

class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                                                                      "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class compas:
    """
    Configuration of dataset Compas
    """

    # the size of total features
    params = 6

    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 38])
    input_bounds.append([0, 20])
    input_bounds.append([1, 10])

    # the name of each feature
    feature_name = ['sex', 'race', 'age_cat', 'priors_count', 'juv_fel_count',
       'r_charge_degree']

    # the name of each class
    class_name = ["no_recid", "recid"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 4, 6, 7, 8, 9, 10]


class meps21:
    """
    Configuration of dataset meps
    """

    # the size of total features
    params = 137



    # the name of each feature
    feature_name = ['age','race','PCS42','MCS42','K6SUM42','REGION=1','REGION=2','REGION=3','REGION=4','sex','MARRY=1','MARRY=2','MARRY=3',
 'MARRY=4','MARRY=5','MARRY=6','MARRY=7','MARRY=8','MARRY=9','MARRY=10','FTSTU=-1','FTSTU=1',
 'FTSTU=2','FTSTU=3','ACTDTY=1','ACTDTY=2','ACTDTY=3','ACTDTY=4','HONRDC=1','HONRDC=2','HONRDC=3','HONRDC=4','RTHLTH=-1',
 'RTHLTH=1','RTHLTH=2','RTHLTH=3','RTHLTH=4','RTHLTH=5','MNHLTH=-1','MNHLTH=1','MNHLTH=2','MNHLTH=3','MNHLTH=4','MNHLTH=5',
 'HIBPDX=-1','HIBPDX=1','HIBPDX=2','CHDDX=-1','CHDDX=1','CHDDX=2','ANGIDX=-1','ANGIDX=1','ANGIDX=2','MIDX=-1','MIDX=1',
 'MIDX=2','OHRTDX=-1','OHRTDX=1','OHRTDX=2','STRKDX=-1','STRKDX=1','STRKDX=2','EMPHDX=-1','EMPHDX=1','EMPHDX=2','CHBRON=-1',
 'CHBRON=1','CHBRON=2','CHOLDX=-1','CHOLDX=1','CHOLDX=2','CANCERDX=-1','CANCERDX=1','CANCERDX=2','DIABDX=-1','DIABDX=1',
 'DIABDX=2','JTPAIN=-1','JTPAIN=1','JTPAIN=2','ARTHDX=-1','ARTHDX=1','ARTHDX=2','ARTHTYPE=-1','ARTHTYPE=1','ARTHTYPE=2',
 'ARTHTYPE=3','ASTHDX=1','ASTHDX=2','ADHDADDX=-1','ADHDADDX=1','ADHDADDX=2','PREGNT=-1','PREGNT=1','PREGNT=2','WLKLIM=-1',
 'WLKLIM=1','WLKLIM=2','ACTLIM=-1','ACTLIM=1','ACTLIM=2','SOCLIM=-1','SOCLIM=1','SOCLIM=2','COGLIM=-1','COGLIM=1','COGLIM=2',
 'DFHEAR42=-1','DFHEAR42=1','DFHEAR42=2','DFSEE42=-1','DFSEE42=1','DFSEE42=2','ADSMOK42=-1','ADSMOK42=1','ADSMOK42=2',
 'PHQ242=-1','PHQ242=0','PHQ242=1','PHQ242=2','PHQ242=3','PHQ242=4','PHQ242=5','PHQ242=6','EMPST=-1','EMPST=1','EMPST=2',
 'EMPST=3','EMPST=4','POVCAT=1','POVCAT=2','POVCAT=3','POVCAT=4','POVCAT=5','INSCOV=1','INSCOV=2','INSCOV=3']

    # the name of each class
    class_name = ["Not utilize", "Utilize"]

    # specify the categorical features with their indices
    categorical_features = [i for i in range(138)]

class credit_random:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    # note that for feature 9 (gender): 1,3,4 are male (replaced with 0) and 2.5 are female (replace with 1)
    params = 20

    input_bounds = []
    input_bounds.append([1, 4])
    input_bounds.append([4, 72])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([250, 18424])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([1, 3])
    input_bounds.append([1, 4])
    input_bounds.append([1, 4])
    input_bounds.append([19, 75])
    input_bounds.append([1, 3])
    input_bounds.append([1, 3])
    input_bounds.append([1, 4])
    input_bounds.append([1, 4])
    input_bounds.append([1, 2])
    input_bounds.append([1, 2])
    input_bounds.append([1, 2])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                    "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]

    
# class law_school:
#     """
#     Configuration of dataset German Credit
#     """

#     # the size of total features
#     # note that for feature 9 (gender): 1,3,4 are male (replaced with 0) and 2.5 are female (replace with 1)
#     params = 13

#     input_bounds = []
#     input_bounds.append([1, 4])
#     input_bounds.append([4, 72])
#     input_bounds.append([0, 4])
#     input_bounds.append([0, 10])
#     input_bounds.append([250, 18424])
#     input_bounds.append([1, 5])
#     input_bounds.append([1, 5])
#     input_bounds.append([1, 4])
#     input_bounds.append([0, 1])
#     input_bounds.append([1, 3])
#     input_bounds.append([1, 4])
#     input_bounds.append([1, 4])
#     input_bounds.append([19, 75])
#     input_bounds.append([1, 3])
    

#     # the name of each feature
#     feature_name = ['sex', 'LSAT', 'UGPA', 'region_first', 'ZFYA', 'sander_index',
#        'Amerindian', 'Asian', 'Black', 'Hispanic', 'Mexican', 'Other',
#        'Puertorican', 'White']

#     # the name of each class
#     class_name = ["bad", "good"]

#     # specify the categorical features with their indices
#     categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


class law:
    """
    Configuration of dataset German Credit
    """

    # the size of total features

    params = 3

    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])


    # the name of each feature
    feature_name = ['race', 'sex', 'LSAT', 'UGPA']

    # the name of each class
    class_name = ["Notpass", "pass"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2]
    
class redcar:
    """
    Configuration of dataset German Credit
    """

    # the size of total features

    params =  6

    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])



    # the name of each feature
    feature_name = ['age', 'race', 'education', 'incom', 'car_type', 'redcar']

    # the name of each class
    class_name = ["No accident", "Accident"]

    # specify the categorical features with their indices

class compas_test:
    """
    Configuration of dataset German Credit
    """

    # the size of total features

    params =  6

    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ['sex','race','age', 'juv_other_count',
           'priors_count','c_charge_degree']
    # the name of each class
    class_name = ["NotRecid", "Recid"]

    # specify the categorical features with thir indices
    categorical_features = [0, 1]

class adult:
    """
    Configuration of dataset German Credit
    """

    # the size of total features

    params =  10

    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ['age', 'workclass', 'educational-num', 'marital-status', 'occupation',
       'relationship', 'race', 'gender', 'hours-per-week', 'native-country']
    # the name of each class
    class_name = ["High", "Low"]

    # specify the categorical features with thir indices
    categorical_features = [0, 1]
class student:
    """
    Configuration of dataset German Credit
    """

    # the size of total features

    params =  5

    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    

    # the name of each feature
    feature_name = ['sex', 'failures', 'higher', 'G1', 'G2']
    # the name of each class
    class_name = ["High", "Low"]

    # specify the categorical features with thir indices
    categorical_features = [0, 1]

