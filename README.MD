## 1. Introduction 
___



### 1.1.Business Understanding
Currently music on-line platform is on the rise. However people still can't fully developed shared their taste. Our Founder feel those and start to develop Music Streaming Platform called **ontrack**. Our Business is based on Subscription Model.
Our Catalogue is Quite Huge contain ~17632 Artists. 


Our business is on decline, previous month we serve nearly 5000 users, however there is significant **churn** leaving our platform only having ~2000 users. 

We are in data science team,asked to assist business development team to tackle this situation. 

### 1.2. Problem Definition 

Business Process 

![Business Process](https://raw.githubusercontent.com/fakhrirobi/artist_recommender/main/assets/Business%20Process.png)

Problem 

    User Churn ~60% !


### 1.3. Business Metrics 
Business Metrics : Churn Rate 

### 1.4. Identifying Machine Learning Problem
Process that could be helped with machine learning : 
![Business Process ](https://raw.githubusercontent.com/fakhrirobi/artist_recommender/main/assets/Business%20Process_1.png)

### 1.5 Objective Metrics 







## 2. Related Work 
___


## 3. Dataset and Features 
___


The dataset obtained from http://www.last.fm, online music system. 
The dataset itself contains : 

1. 2K Users listening activity 



## 4. Methods
___


### 4.1. Recommender System Introduction

Recomender system
Problem 

We have implicit feedback dataset that reflects number of user plays from certain artist.

Given those utility matrix we will recommend to each user what artist they might like. 


The recommendation will be in form such as 
1. "Since you listen to artist X" --> give list 
2. etc. 

Since we have enough utility matrix / implicit feedback we will start with collaborative filtering approach. 

Collaborative filtering approach can be divided into several approaches 

![Branching in Collaborative Filtering](https://raw.githubusercontent.com/fakhrirobi/artist_recommender/main/assets/Collaborative%20Filtering%20Branching.png)


### 4.2.Baseline (Popularity Recommendation)

### 4.3. Alternating Least Square (Implicit Feedback Matrix Factorization)
Hu, Yifan, Yehuda Koren, and Chris Volinsky. 2008. “Collaborative Filtering for Implicit Feedback Datasets.” In 2008 Eighth IEEE International Conference on Data Mining, 263–72. IEEE.


### 4.3. Bayesian Personalized Ranking 
Rendle, Steffen, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. “BPR: Bayesian Personalized Ranking from Implicit Feedback.” In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, 452–61. AUAI Press.


### 4.4. Logistic Matrix Factorization
Johnson, Christopher C. 2014. “Logistic Matrix Factorization for Implicit Feedback Data.” Advances in Neural Information Processing Systems 27.






## 5.Experiments / Results / Discussion 
___
### 5.1. Data Splitting Strategy 
![Splitting](https://raw.githubusercontent.com/fakhrirobi/artist_recommender/main/assets/Splitting.png)
### 5.2. Data Preprocessing 


#### 5.2.1 Mapping  `userID` & `artistsID` into Ordered ID
Our data requirement is in utility matrix form, it would be hard to access each element in utility matrix since utility matrix has ordered id. We need to create mapping for both `userID, artistsID to orderedID` and vice versa.Our Mapping is in python dictionary object. 
```

user_id_to_ordered_id = {
    userID:ordereduserID
}

example : 
user_id_to_ordered_id = {
    2:1
}


```
for later usage (such as : API) we will serialize object using `joblib.dump` function as pickle file (`.pkl`), named : 
1. `user_id_to_ordered_id.pkl`
2. `ordered_id_to_user_id.pkl`
3. `artist_id_to_ordered_id.pkl`
4. `ordered_id_to_artist_id.pkl`





### 5.3. Model Selection 
Result : 

| No | precision @10 |   map@10 |  ndcg@10 |   auc@10 |                       model |   |   |
|---:|--------------:|---------:|---------:|---------:|----------------------------:|---|---|
|  1 |      0.135773 | 0.062366 | 0.134182 | 0.565975 |     AlternatingLeastSquares |   |   |
|  2 |      0.112814 | 0.053603 | 0.115200 | 0.553911 | BayesianPersonalizedRanking |   |   |
|  3 |      0.014199 | 0.004576 | 0.013120 | 0.506782 | LogisticMatrixFactorization |   |   |
|    |               |          |          |          |                             |   |   |

from several metrics above we can see that `AlternatingLeastSquares` outperform other models --> move to Hyperparameter Tuning Phase 

### 5.4. Hyperparameter Tuning
In this process we will find the best parameters pair for our best selected models,`AlternatingLeastSquares`. 

Hyperparameters that are available in  `AlternatingLeastSquares` models are : 

1. `factors`
   
   number of latent factors, commonly found in matrix factorization model, including `AlternatingLeastSquares`, for this hyperparameter we will pick candidate value `[100,200,300,400,500]`
   
2. `alpha` 
   
   In our  `AlternatingLeastSquares` model we have weight   `alpha` as confidence magnitude from user implicit interactions such as number of clicks, etc. values from alpha we would like to choose, ranging from `0.01` to `1.0`. 
   
3. `regularization` 
   
   number of how strong we imposed regularization on weights, this due to the sparsity of the data, we dont want the weight become so big --> prone to overfitting. For this hyperparameter we will try some values ranging from `0.01` to `0.2`

Some approach in hyperparameter tuning : 

1.  **GridSearch** 
   
   This approach simply fit all combinations from hyperparameter candidate, let say we have 3 hyperparameter with each candidate value of 3, number of model fitting is 3*3*3 = 27 times fitting.This is not **efficient** approach, especially with recommender system model with huge size of data. 

2.  **RandomizedSearch**
   
   This approach perform better than **GridSearch** approach in terms of computation because it only samples / subset from our hyperparameter. 

   ![Grid & Randomized](https://miro.medium.com/v2/1*ZTlQm_WRcrNqL-nLnx6GJA.png)

3.  **Bayesian Optimization**
   
    This approach hyperparameter value as Gaussian Process problem where the hyperparemeter value is the product of Surrogate function (such as Gaussian Process), we will use this approach bcause it efficient in terms of computation and provide better result. 
    Don't worry we don't have to understand all right now, and we will not coding it from scratch, we will use [**optuna**](https://optuna.org/) package for now. 

For hyperparameter set up we will perform Cross-Validation with --> **K-Fold Cross Validation**

![KFold](data:image/png;base64,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)

Best parameters  : 
1. `factors` : **100**
2. `alpha` : **0.5097051938957499**
3. `regularization` : **0.16799704422342204**

Final metrics : 
{'precision': 0.16635468872652617,
 'map': 0.0844339689737369,
 'ndcg': 0.17261162377361844,
 'auc': 0.574831593418623}
### 5.5 Evaluation 
On final evaluation we measured tuned model on test set, the result 

|       precision @10 |             map@10 |             ndcg@10 |            auc@10 |   |   |
|--------------------:|-------------------:|--------------------:|------------------:|---|---|
| 0.16635468872652617 | 0.0844339689737369 | 0.17261162377361844 | 0.574831593418623 |   |   |
|                     |                    |                     |                   |   |   |

### 5.6. Sanity Check on Recommendation



## 6.Conclusion 
___

### 6.1. Further Work 
1. Using more user oriented metrics such as --> Diversity, Serendipity, Novelty
2. Using Multistage Approach 
3. Apply Graph Data (Friends)
4. Use Metadata as Features --> For Using Factorization Machine
5. 



5. A 
6. B
7. C
   