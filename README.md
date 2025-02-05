# `hyperopt` Demo: Travel Insurance Claims Prediction

Suppose we're working with a travel insurance data to predict whether an insurant will file a claim. This seemingly straightforward classification problem actually presents several challenging aspects that make it perfect for demonstrating the importance of proper hyperparameter tuning.

![image](https://github.com/user-attachments/assets/b205f816-ab9c-48ab-86a9-5b685fd4f314)

Our dataset contains various features about insurance customers, but it comes with a common challenge in insurance data: class imbalance. The vast majority of customers don't file claims, making our target variable highly skewed. This imbalance creates a unique challenge for our model - **it needs to learn patterns from a small number of positive cases (customers who filed claims) while not being overwhelmed by the abundance of negative cases**.

![image](https://github.com/user-attachments/assets/6a5f0ce9-eae5-47ce-9b7c-fafd030c17e3)

We've handled outliers in our initial setup. Our preprocessing pipeline includes proper handling of both categorical and numerical features, and we've carefully split our data into training and test sets using stratification to maintain the same class distribution across both sets. For our model, we've chosen XGBoost (`XGBClassifier`), a powerful gradient boosting algorithm that often performs well on kaggle competitions.


To address the class imbalance, we've made some initial adjustments to our XGBoost model. We've set `max_delta_step=1` to help stabilize the updates in our boosting process, and we've used `scale_pos_weight` to give more importance to our minority class (customers who filed claims). This weight is is set to be the ratio between non-claim and claim instances.

![image](https://github.com/user-attachments/assets/9b479efb-f20a-4017-b56e-72fbaf3ed517)


However, despite our preparations, our model isn't performing as well as we'd hope. We're particularly interested in minimizing false negatives - cases where we predict a customer won't file a claim but they actually do. This makes recall our key metric. However, our results are concerning: **while our model achieves nearly perfect recall on the training data (close to 1.0), it drops dramatically to around 0.4 on our test data. This substantial gap between training and test performance is a classic sign of overfitting** - our model has learned the training data too well but fails to generalize to new cases. This suggests that **the default values for other parameters of `XGBClassifier` are unsuited for imbalanced data**.

![image](https://github.com/user-attachments/assets/1e1ecbd2-81f7-499e-b158-a0187208cd79)

## Hyperopt to the Rescue
Now that we've identified our model's overfitting problem, we'll configure Hyperopt's search space to address this issue. Our approach focuses on constraining parameters that typically influence model complexity. 


Let's examine how we've crafted our parameter ranges to combat overfitting. For tree-related parameters like `max_depth`, we're keeping values relatively low (between 3 and 12) compared to XGBoost's default of unlimited depth. Similarly, we're exploring modest values for `n_estimators` (50-100. Default value: 100) and `learning_rate` (0.01-0.2. Default value: 0.3) to prevent the model from becoming too complex too quickly.

For `min_child_weight`, we're setting a higher range (10-40) which requires more observations in each tree node, making it harder for the model to learn noise in the training data. This parameter is particularly important in our case as it helps prevent the model from overfitting to rare patterns in our imbalanced dataset.

We're using high value range for `scale_pos_weight` (40-80) because our dataset is heavily imbalanced. The exact optimal value may not correspond  to the ratio of negative to positive examples in the dataset, as can be seen in the untuned model.

`max_delta_step` (set to 1, 3, or 5) works as a safety mechanism alongside `scale_pos_weight`. While `scale_pos_weight` increases attention on claim cases, `max_delta_step` prevents the model from making too-dramatic corrections in response. Keeping it low ensures the model adjusts its predictions gradually and stably, even when dealing with rare claim cases.


And so, here's our carefully crafted parameter space definition:

```python
params_space = {
        'max_depth' : hp.choice('max_depth',range(3, 12, 1)), # low values to reduce overfitting
        'n_estimators' : scope.int(hp.quniform("n_estimators", 50, 100,5)), # low values to reduce overfitting
        'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), # low values to reduce overfitting
        'min_child_weight': scope.int(hp.quniform("min_child_weight", 10, 40, 1)), # high values to reduce overfitting
        'scale_pos_weight': hp.uniform("scale_pos_weight", 40, 80), # high values because data is highly imbalanced
        'max_delta_step': hp.choice("max_delta_step", [1,3,5]), # low values to reduce overfitting
        'subsample': hp.choice("subsample", [0.5,0.6, 0.7, 0.8, 0.9, 1.0]), # values between 0.5-1.0 to reduce overfitting
        'colsample_bytree': hp.choice("colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # values between 0.5-1.0 to reduce overfitting
    }
```

Next, we define our objective function that Hyperopt will optimize. This function will pass the tuned parameters into the model, perform cross-validation, and return the loss score (1 minus the mean recall):

```python
# Objective function to be minimized by hyperopt
def objective(args):
    
    model_pipe = Pipeline([
        ('preprocessing', transformer),
        ('model', XGBClassifier(random_state=0, **args))])
    
    skfold = StratifiedKFold(n_splits = 5, shuffle=True,random_state=0)
    model_cv = cross_val_score(
        model_pipe, 
        X_train, 
        y_train, 
        cv = skfold, 
        scoring = 'recall',
        )
    
    loss_score = 1 - model_cv.mean() # Set to 1-recall because hyperopt minimizes

    return {'loss': loss_score, 'status': STATUS_OK}

# Initiate a Trial object 
trials = Trials()

# Iterate with fmin to get best parameters
tuning_result = fmin(fn=objective, 
                     space=params_space, 
                     algo=tpe.suggest,
                     max_evals=50, 
                     trials=trials, 
                     rstate=np.random.default_rng(0))

# Obtain best parameters from tuning_result (an object) and params_space
best_params = space_eval(params_space, tuning_result)
```

![image](https://github.com/user-attachments/assets/f3c9d3e2-402e-4bdd-96a3-f45540bd9559)

**As shown in the accompanying image, Hyperopt found an optimal set of parameters in just 50 trials**. But the real proof of success lies in the tuned model's performance metrics (see below). **The recall on our test dataset has skyrocketed from 41% to 82%**. Note also how the gap between training and test recall has narrowed significantly, indicating that our model is now generalizing much better to unseen data. To achieve similar results using Grid Search, we could have needed to evaluate many more combinations, consuming significantly more computational resources and time.

![image](https://github.com/user-attachments/assets/517523f0-5b3e-46bb-a2f5-325d26ee546e)

## Conclusion
As we've seen in our travel insurance example, Hyperopt offers a sophisticated yet accessible approach to finding optimal model parameters. By leveraging Bayesian optimization through the TPE algorithm, Hyperopt efficiently navigates complex parameter spaces that would be computationally prohibitive for grid search methods. This efficiency showcases why Hyperopt has become such a valuable tool in the contemporary data science toolkit.

---
**References**
1. https://xgboosting.com/
2. http://hyperopt.github.io/hyperopt/
