setwd('C:/Users/BTHANISH/Desktop/Thanish/Competition/Crowd Analytix')

library(data.table)
library(MLmetrics)
library(ModelMetrics)

train_men = fread('mens_train_file.csv', stringsAsFactors = T)
test_men = fread('mens_test_file.csv', stringsAsFactors = T)
train_women = fread('womens_train_file.csv', stringsAsFactors = T)
test_women = fread('womens_test_file.csv', stringsAsFactors = T)

sample_sub = read.csv('AUS_SubmissionFormat.csv')
sub_id_sort = factor(sample_sub$submission_id, levels = sample_sub$submission_id)

#Appending all data set
train_test_prod = rbind(train_men, test_men, train_women, test_women)

#Forming the submission ID
train_test_prod[, sub_id := paste(id, gender, sep= '_')]

#Convert logical columns to numeric
logi_columns = colnames(train_test_prod)[sapply(train_test_prod, FUN = class) == 'logical']
train_test_prod[,(logi_columns):= lapply(.SD, as.numeric), .SDcols = logi_columns]

#Splitting in to train and test
train_prod = train_test_prod[!is.na(outcome), ]
test_prod = train_test_prod[is.na(outcome), ]

#Drop off the not used columns
test_sub_id = test_prod$sub_id
train_prod[, ':=' (id = NULL, 
                   train = NULL,
                   #net_clr_bins = NULL,
                   #gender = NULL,
                   sub_id = NULL)]
test_prod[, ':=' (id = NULL, 
                  train = NULL,
                  #net_clr_bins = NULL,
                  #gender = NULL,
                  sub_id = NULL)]


library(caret)
set.seed(100)
folds = createFolds(train_prod$outcome, k = 5)

library(h2o)
h2o.init(nthreads = -1, min_mem_size = '20g')
train_prod_h2o = as.h2o(train_prod)
test_prod_h2o = as.h2o(test_prod)

x_indep = setdiff(colnames(train_prod_h2o), c('outcome'))
y_dep = 'outcome'

########################################################################################################
#Forming the 1st stack data
stack_train_fold_GBM = data.table()
stack_test_fold_GBM = data.table(FE_GBM_1 = rep(0,nrow(test_prod)), 
                             UE_GBM_1 = rep(0,nrow(test_prod)), 
                             W_GBM_1 = rep(0,nrow(test_prod)))
for (i in 1: length(folds))
{
  train_prod_h2o_folds = train_prod_h2o[-folds[[i]],]
  test_h2o_folds = train_prod_h2o[folds[[i]],]
  
  ########################## gradient Boosting
  GBM.h2o.model = h2o.gbm(x = x_indep, y = y_dep, training_frame = train_prod_h2o_folds,
                                  ntrees = 100, max_depth = 5, learn_rate = 0.1,
                                  #nfolds = 5, fold_assignment = 'Stratified', 
                                  seed = 100)
  GBM.h2o.pred.fold = h2o.predict(GBM.h2o.model, newdata = test_h2o_folds)
  GBM.h2o.pred.fold = as.data.frame(GBM.h2o.pred.fold)
  
  loss = mlogLoss(actual = train_prod$outcome[folds[[i]]], predicted = GBM.h2o.pred.fold[,c('FE', 'UE', 'W')])
  print(paste('Log_loss of fold for GBM', i, 'is', loss))
  
  #Forming the stack train prod
  GBM.h2o.pred.fold$outcome = train_prod$outcome[folds[[i]]]
  stack_train_fold_GBM = rbind(stack_train_fold_GBM, GBM.h2o.pred.fold[,c('FE', 'UE', 'W', 'outcome')])
  
  #Forming the stack test prod
  GBM.h2o.pred = h2o.predict(GBM.h2o.model, newdata = test_prod_h2o)
  GBM.h2o.pred = as.data.frame(GBM.h2o.pred)
  #print(head(GBM.h2o.pred))
  stack_test_fold_GBM$FE_GBM_1 = stack_test_fold_GBM$FE_GBM_1 + GBM.h2o.pred$FE
  stack_test_fold_GBM$UE_GBM_1 = stack_test_fold_GBM$UE_GBM_1 + GBM.h2o.pred$UE
  stack_test_fold_GBM$W_GBM_1  = stack_test_fold_GBM$W_GBM_1  + GBM.h2o.pred$W
  
  
  #stack_test_fold_GBM = rbind(stack_test_fold_GBM, RF.h2o.pred)
  if (i == length(folds))
  {
    colnames(stack_train_fold_GBM) = c('FE_GBM_1', 'UE_GBM_1', 'W_GBM_1', 'outcome')
    stack_test_fold_GBM = stack_test_fold_GBM/i
  }
}

########################################################################################################

#Forming the 2nd stack data
stack_train_fold_RF = data.table()
stack_test_fold_RF = data.table(FE_RF_1 = rep(0,nrow(test_prod)), 
                             UE_RF_1 = rep(0,nrow(test_prod)), 
                             W_RF_1 = rep(0,nrow(test_prod)))
for (i in 1: length(folds))
{
  train_prod_h2o_folds = train_prod_h2o[-folds[[i]],]
  test_h2o_folds = train_prod_h2o[folds[[i]],]
  
  ########################## gradient Boosting
  RF.h2o.model = h2o.randomForest(x = x_indep, y = y_dep, training_frame = train_prod_h2o_folds,
                                  ntrees = 200, max_depth = 15, mtries = 11,
                                  #nfolds = 5, fold_assignment = 'Stratified', 
                                  seed = 100)
  RF.h2o.pred.fold = h2o.predict(RF.h2o.model, newdata = test_h2o_folds)
  RF.h2o.pred.fold = as.data.frame(RF.h2o.pred.fold)
  
  loss = mlogLoss(actual = train_prod$outcome[folds[[i]]], predicted = RF.h2o.pred.fold[,c('FE', 'UE', 'W')])
  print(paste('Log_loss of fold for random Forest', i, 'is', loss))
  
  #Forming the stack train prod
  RF.h2o.pred.fold$outcome = train_prod$outcome[folds[[i]]]
  stack_train_fold_RF = rbind(stack_train_fold_RF, RF.h2o.pred.fold[,c('FE', 'UE', 'W')])
  
  #Forming the stack test prod
  RF.h2o.pred = h2o.predict(RF.h2o.model, newdata = test_prod_h2o)
  RF.h2o.pred = as.data.frame(RF.h2o.pred)
  #print(head(RF.h2o.pred))
  stack_test_fold_RF$FE_RF_1 = stack_test_fold_RF$FE_RF_1 + RF.h2o.pred$FE
  stack_test_fold_RF$UE_RF_1 = stack_test_fold_RF$UE_RF_1 + RF.h2o.pred$UE
  stack_test_fold_RF$W_RF_1  = stack_test_fold_RF$W_RF_1  + RF.h2o.pred$W
  
  
  #stack_test_fold_RF = rbind(stack_test_fold_RF, RF.h2o.pred)
  if (i == length(folds))
  {
    colnames(stack_train_fold_RF) = c('FE_RF_1', 'UE_RF_1', 'W_RF_1')
    stack_test_fold_RF = stack_test_fold_RF/i
  }
}

########################################################################################################
library(xgboost)

fac_columns = colnames(train_prod)[sapply(train_prod, FUN = class) == 'factor']

train_prod[,(fac_columns):= lapply(.SD, as.numeric), .SDcols = fac_columns]
test_prod[,(fac_columns):= lapply(.SD, as.numeric), .SDcols = fac_columns]

train_prod[, outcome := outcome - 1]

x_indep = setdiff(colnames(train_prod), 'outcome')
y_dep = 'outcome'

stack_train_fold_XGB = data.table()
stack_test_fold_XGB = data.table(FE_XGB_1 = rep(0,nrow(test_prod)), 
                                 UE_XGB_1 = rep(0,nrow(test_prod)), 
                                 W_XGB_1 = rep(0,nrow(test_prod)))
for (i in 1: length(folds))
{
  train_prod_folds = train_prod[-folds[[i]],]
  test_folds = train_prod[folds[[i]],]
  
  dtrain_prod_fold = xgb.DMatrix(data = as.matrix(train_prod_folds[,x_indep, with = F]), label = train_prod_folds$outcome)
  dtest_prod_fold = xgb.DMatrix(data = as.matrix(test_folds[,x_indep, with = F]), label = test_folds$outcome)
  dtest_prod =  xgb.DMatrix(data = as.matrix(test_prod[, x_indep, with = F]))
  
  watchlist = list(train = dtrain_prod_fold, test = dtest_prod_fold)
  
  set.seed(100)
  xgb.model.prod = xgb.train(data = dtrain_prod_fold, 
                             watchlist = watchlist,
                             nrounds = 1000,
                             eta = 0.1,
                             max_depth = 6,
                             objective = 'multi:softprob',
                             num_class = 3,
                             subsample = 0.8,
                             colsample_bytree = 0.9,
                             eval_metric = 'mlogloss',
                             min_child_weight = 1,
                             early_stopping_rounds = 10
                             )
  
  xgb.pred.fold = predict(xgb.model.prod, newdata = dtest_prod_fold)
  xgb.pred.fold = as.data.frame(matrix(xgb.pred.fold, nrow = nrow(test_folds), byrow = T))
  colnames(xgb.pred.fold) = c('FE', 'UE', 'W')
  xgb.pred.fold = xgb.pred.fold[, c('UE', 'FE', 'W')]
  
  #Forming the stack train prod
  stack_train_fold_XGB = rbind(stack_train_fold_XGB, xgb.pred.fold[,c('FE', 'UE', 'W')])
  
  #Forming the stack test prod
  xgb.pred.prod = predict(xgb.model.prod, newdata = dtest_prod)
  xgb.pred.prod = as.data.frame(matrix(xgb.pred.prod, nrow = nrow(test_prod), byrow = T))
  colnames(xgb.pred.prod) = c('FE', 'UE', 'W')
  xgb.pred.prod = xgb.pred.prod[, c('UE', 'FE', 'W')]
  
  #print(head(XGB.h2o.pred))
  stack_test_fold_XGB$FE_XGB_1 = stack_test_fold_XGB$FE_XGB_1 + xgb.pred.prod$FE
  stack_test_fold_XGB$UE_XGB_1 = stack_test_fold_XGB$UE_XGB_1 + xgb.pred.prod$UE
  stack_test_fold_XGB$W_XGB_1  = stack_test_fold_XGB$W_XGB_1  + xgb.pred.prod$W
  
  
  #stack_test_fold_XGB = rbind(stack_test_fold_XGB, XGB.h2o.pred)
  if (i == length(folds))
  {
    colnames(stack_train_fold_XGB) = c('FE_XGB_1', 'UE_XGB_1', 'W_XGB_1')
    stack_test_fold_XGB = stack_test_fold_XGB/i
  }
  
}


########################################################################################################

stack_train_fold = cbind(stack_train_fold_GBM, stack_train_fold_RF, stack_train_fold_XGB)
stack_test_fold = cbind(stack_test_fold_GBM, stack_test_fold_RF, stack_test_fold_XGB)

##############################################################################
#Used the stacked data to create model
##############################################################################
x_indep = setdiff(colnames(stack_train_fold), c('outcome'))
y_dep = 'outcome'

stack_train_fold[, outcome := outcome - 1]

dtrain_prod <- lgb.Dataset(data = as.matrix(stack_train_fold[, x_indep, with = F]), label = stack_train_fold$outcome)
dtest_prod <- lgb.Dataset(data = as.matrix(stack_test_fold[, x_indep, with = F]))

valids <- list(train = dtrain_prod, test = dtest_prod)

params <- list(objective = "multiclass"
               ,metric = "multi_logloss" 
               ,num_class = 3
               , min_data_in_leaf = 1
               #,max_depth = 2
               #, feature_fraction = 0.9
               #, bagging_fraction = 0.6
               #, bagging_freq = 1
)

set.seed(100)
lgb.model.cv = lgb.cv(params = params,
                      data = dtrain_prod,
                      nfold = 5,
                      #num_leaves = 100,
                      stratified = T,
                      early_stopping_rounds = 10,
                      nrounds = 1000,
                      learning_rate = 0.05
)
lgb.model.cv$best_score

set.seed(100)
lgb.model.prod <- lgb.train(params = params,
                            data = dtrain_prod,
                            nrounds = lgb.model.cv$best_iter,
                            valids = valids,
                            learning_rate = 0.05
                            #,early_stopping_rounds = 10
)


lgb.prod.pred <- predict(lgb.model.prod, as.matrix(stack_test_fold), reshape = TRUE)
head(lgb.prod.pred)

colnames(lgb.prod.pred) = c('FE', 'UE', 'W')
lgb.prod.pred = lgb.prod.pred[, c('UE', 'FE', 'W')]
head(lgb.prod.pred)

#Submission file
sub_lgb = cbind(data.frame(submission_id = test_sub_id,	train = 0), lgb.prod.pred)
head(sub_lgb)

sub_lgb$submission_id = factor(sub_lgb$submission_id, levels = sub_id_sort)
sub_lgb = sub_lgb[order(sub_lgb$submission_id),]
head(sub_lgb)

write.csv(sub_lgb, row.names = F, quote = F,'sub_stack_en_3.csv')
