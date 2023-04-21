deeper:Deep Ensemble for Environmental Predictor
================
Alven Yu
2022/05/20

-   [Installation](#installation)
-   [Steps](#steps)
    -   [\* Step 1: establish the base
        models](#-step-1-establish-the-base-models)
    -   [\* Step 2: stack the meta
        models](#-step-2-stack-the-meta-models)
    -   [\* Step 3: prediction based on the new data
        set](#-step-3-prediction-based-on-the-new-data-set)
-   [Example](#example)
    -   [1. Data preparation](#1-data-preparation)
        -   [Load the example data](#load-the-example-data)
        -   [separate it to training and testing
            dataset](#separate-it-to-training-and-testing-dataset)
        -   [Identify the dependence and independence
            variables](#identify-the-dependence-and-independence-variables)
    -   [2.The models can be used](#2the-models-can-be-used)
    -   [3.Model Building](#3model-building)
    -   [3.1 Tuning the paramaters of a base
        model](#31-tuning-the-paramaters-of-a-base-model)
    -   [3.2 Establish DEML model](#32-establish-deml-model)
        -   [3.2.1 Training the (base) ensemble
            model](#321-training-the-base-ensemble-model)
        -   [3.2.2 Training the (base) ensemble model with parallel
            computing](#322-training-the-base-ensemble-model-with-parallel-computing)
    -   [3.3 Stacked meta models](#33-stacked-meta-models)
    -   [4. Assess and plot the results](#4-assess-and-plot-the-results)
    -   [Citation](#citation)

------------------------------------------------------------------------

## Installation

You can install the developing version of deeper from
[github](https://github.com/Alven8816/deeper) with:

``` r
library(devtools)
install_github("Alven8816/deeper")
```

------------------------------------------------------------------------

## Steps

    Our newly build R package mainly include 3 steps:

### \* Step 1: establish the base models

Using predictModel() or predictModel\_parallel() to establish the base
models. A tuningModel() function can be used to tuning the parameters to
get the best single base model.

### \* Step 2: stack the meta models

We use stack\_ensemble(),stack\_ensemble.fit(), or
stack\_ensemble\_parallel() function to stack the meta models to get a
DEML model.

### \* Step 3: prediction based on the new data set

After establishment of DEML model, the predict() can be used predict the
unseen data set.

To assess the performance of the models, assess.plot() can be used by
comparing the original observations (y in test set) and the prediction.
The assess.plot() also return the point scatter plot.

The other functions including CV.predictModel() (referred from
SuperLearner), CV.predictModel\_parallel(),and
CV.stack\_ensemble\_parallel() could be used to conduct the external
cross validation to assess the base models and DEML models for each
fold.

**Note: DEML could not directly deal with missing values and missing
value imputation technologies is recommended prior to the use of the
DEML model.**

------------------------------------------------------------------------

# Example

This is a basic example which shows you how to use deeper:

## 1. Data preparation

### Load the example data

``` r
library(deeper)

## obtain the example data
data("envir_example")

#knitr::kable(envir_example[1:6,])
```

``` r
summary(envir_example)
```

### separate it to training and testing dataset

    80% data was used as training and the reminding as testing.

``` r
set.seed(1234)
size <-
  caret::createDataPartition(y = envir_example$PM2.5, p = 0.8, list = FALSE)
trainset <- envir_example[size, ]
testset <- envir_example[-size, ]
```

### Identify the dependence and independence variables

``` r
y <- c("PM2.5")
x <- colnames(envir_example[-c(1, 6)]) # except "date" and "PM2.5"
```

## 2.The models can be used

``` r
# we need to select the algorithm previously
SuperLearner::listWrappers()
data(model_list) # get more details about the algorithm
```

In the ‘type’ column, “R”: can be used for regression or
classification;“N”: can be used for regression but variables requre to
be numeric style; “C”: just used in classification.

## 3.Model Building

## 3.1 Tuning the paramaters of a base model

``` r
#method 1: using deeper package function
ranger <-
  tuningModel(
    basemodel  = 'SL.ranger',
    params = list(num.trees = 100),
    tune = list(mtry = c(1, 3, 7))
  )

#method 2: employ the Superlearner package "create.Learner" method
#ranger <- SuperLearner::create.Learner(base_learner = 'SL.ranger',params = list(num.trees = 1000),tune = list(mtry = c(1,3,7)))

#method 3: create a function by oneself as follow
#ranger <- function(...){SL.ranger(...,num.trees = 1000)}
```

## 3.2 Establish DEML model

### 3.2.1 Training the (base) ensemble model

``` r
# training the model with separate new data set

pred_m2 <-
  predictModel(
    Y = trainset[, y],
    X = trainset[, x],
    base_model = c("SL.xgboost", ranger),
    cvControl = list(V = 5)
  )

# predict the new dataset
pred_m2_new <- deeper::predict(object = pred_m2, newX = testset[, x])

# the base model prediction
head(pred_m2_new$pre_base$library.predict)
```

The results show the weight, R2, and the root-mean-square error (RMSE)
of each model. “ranger\_1”,“ranger\_2”,“ranger\_3” note the Random
Forest model with parameters mtry = 1,3,7 separately.

The results show that mtry = 5 is the best RF model for this
prediction.This method could be used to tune the suitable parameters for
algorithms.

### 3.2.2 Training the (base) ensemble model with parallel computing

``` r
## conduct the spatial CV

# Create a list with 7 (folds) elements (each element contains index of rows to be considered on each fold)

indices <-
  CAST::CreateSpacetimeFolds(trainset, spacevar = "code", k = 7)

# Rows of validation set on each fold

v_raw <- indices$indexOut

names(v_raw) <- seq(1:7)

pred_m3 <- predictModel_parallel(
  Y = trainset[, y],
  X = trainset[, x],
  base_model = c("SL.xgboost", ranger),
  cvControl = list(V = length(v_raw), validRows = v_raw),
  number_cores = 4,
  seed = 1
)
## when number_cores is missing, it will indicate user to set one based on the operation system.

# pred_m3 <- predictModel_parallel(
#     Y = trainset[,y],
#     X = trainset[,x],
#     base_model = c("SL.xgboost",ranger),
#     cvControl = list(V = length(v_raw), validRows = v_raw),
#     seed = 1
#   )

#You have 8 cpu cores, How many cpu core you want to use:
# type the number to continue the process.


# prediction
pred_m3_new <- deeper::predict(object = pred_m3, newX = testset[, x])

head(pred_m3_new$pre_base$library.predict)
```

``` r
# test dataset performance
test_p1 <-
  assess.plot(pre = pred_m3_new$pre_base$pred, obs = testset[, y])

print(test_p1$plot)
```

## 3.3 Stacked meta models

    Our DEML model includes several meta-models based on the trained base models' results. Different from other ensemble (stacking method), DEML model allow to select several meta-models and run them parallelly and ensemble all the meta-models with the optimal weights.(This step is optinal for analysis).

``` r
#Include original feature
pred_stack_10 <-
  stack_ensemble_parallel(
    object = pred_m3,
    Y = trainset[, y],
    meta_model = c("SL.ranger", "SL.xgboost", "SL.glmnet"),
    original_feature = TRUE,
    X = trainset[, x],
    number_cores = 4
  )

pred_stack_10_new <-
  predict(object = pred_stack_10, newX = testset[, x])
```

## 4. Assess and plot the results

``` r
test_p2 <-
  assess.plot(pre = pred_stack_10_new$pre_meta$pred, obs = testset[, y])

print(test_p2$plot)
```

## Citation

Wenhua Yu, Shanshan Li, Tingting Ye,Rongbin Xu, Jiangning Song, Yuming
Guo (2022) Deep ensemble machine learning framework for the estimation
of PM2.5 concentrations,Environmental health perspectives:
<https://doi.org/10.1289/EHP9752>
