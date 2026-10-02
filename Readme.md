
Features for the project: 
    Age
    NumOfProducts
    IsActiveMember
    Geography
    Gender
    Balance
    CreditScore
    Tenure
    HasCrCard
    EstimatedSalary
    OneProduct (1 = 1 Product)
    ManyProducts (1 = 3-4 Products)
    ZeroBalance
    AgeXInactive (1 = old and inactive) 
    IsSynthetic (1 = Kaggle 0=Original_data )


olde_age = 40 (suggestion Leo, because there the exit rate jumps from 6% of over 30% )



I left the Churn_Modelling.csv out of my code for now to compare the models more easily, but it might improve the results, so I would suggest that we try it with the original data as well.

At the moment I have only built simple versions of HistGB (no tuning so far) and LogReg (which is already well calibrated from the start). I will start testing now and see where I end up.