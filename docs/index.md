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

Problem :

User Churn ~60% !


### 1.3. Business Metrics 
Business Metrics : Churn Rate 

### 1.4. Identifying Machine Learning Problem
Process that could be helped with machine learning : 
![Business Process ](https://raw.githubusercontent.com/fakhrirobi/artist_recommender/main/assets/Business%20Process_1.png)

Possible Solutions : 

| No |                                              Solution |           Task | Detailed Task         | Metrics                                                |
|---:|------------------------------------------------------:|---------------:|-----------------------|--------------------------------------------------------|
| 1  | Predict number of user play count on each artist      | Regression     | Count Regression      | Prediction Error (RMSE,MAE,etc)                        |
| 2  | Predict whether user will like the artist or not      | Classification | Binary Classification | Decision Support Metrics (Precision,Recall,AUC,F1,etc) |
| 3  | Predict condfidence scale (0-1) how user like an item | Regression     | -                     | Prediction Error (RMSE,MAE,etc)                        |
| 4  | Predict the ranking from user to artists              | Ranking        | Pairwise Ranking      | Ranking Metrics (ex : NDCG,MAP,MRR,etc)                |

### 1.5 Objective Metrics 






## 2. Related Work 
___
1. Hu, Y., Ogihara, M.: Nextone player: A music recommendation system based on user
behaviour. In: Int. Society for Music Information Retrieval Conf. (ISMIR’11) (2011)

2. Hariri, N., Mobasher, B., Burke, R.: Context-aware music recommendation based on latenttopic sequential patterns. In: Proceedings of the sixth ACM conference on Recommender
systems, pp. 131–138. ACM (2012)

## 3. Dataset and Features 
___


The dataset obtained from http://www.last.fm, online music system. 
The dataset itself contains several files 

1. `artists.dat`
   
   Contains features : 

    - `name`
    - `url`
    - `pictureURL`
  

2. `tags.dat`
      
   Contains features : 

    - `tagID`
    - `tagValue`
  
3. `user_artists.dat`
   
   Contains features : 

    - `userID`
    - `artistsID`
    - `pictureURL`
  
4. `user_friends.dat`
      
   Contains features : 

    - `name`
    - `url`
    - `pictureURL`
  
5. `user_taggedartists.dat`
      
   Contains features : 

    - `name`
    - `url`
    - `pictureURL`



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

| No | precision @10 |   map@10 |  ndcg@10 |   auc@10 |                       model |
|---:|--------------:|---------:|---------:|---------:|----------------------------:|
|  1 |      0.135773 | 0.062366 | 0.134182 | 0.565975 |     AlternatingLeastSquares | 
|  2 |      0.112814 | 0.053603 | 0.115200 | 0.553911 | BayesianPersonalizedRanking | 
|  3 |      0.014199 | 0.004576 | 0.013120 | 0.506782 | LogisticMatrixFactorization | 


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

![KFold](https://zitaoshen.rbind.io/project/machine_learning/machine-learning-101-cross-vaildation/featured.png) [source](https://zitaoshen.rbind.io/project/machine_learning/machine-learning-101-cross-vaildation/featured.png)


Best parameters  : 
1. `factors` : **100**
2. `alpha` : **0.5097051938957499**
3. `regularization` : **0.16799704422342204**


### 5.5 Evaluation 

On final evaluation we measured tuned model on test set, the result 

|       precision @10 |             map@10 |             ndcg@10 |            auc@10 |
|--------------------:|-------------------:|--------------------:|------------------:|
| 0.16635468872652617 | 0.0844339689737369 | 0.17261162377361844 | 0.574831593418623 |


### 5.6. Sanity Check on Recommendation



## 6.Conclusion 
___

### 6.1. Further Work 
1. Using more user oriented metrics such as  **Diversity, Serendipity, Novelty**
2. Using Multistage Approach 
3. Apply Graph Data (Friends)
4. Use Metadata as Features,for Using Factorization Machine


## 7. Product 
___

### 7.1. API 
![APIS](https://cdn.ttgtmedia.com/rms/onlineimages/how_an_api_works-f_mobile.png)


[src](https://cdn.ttgtmedia.com/rms/onlineimages/how_an_api_works-f_mobile.png)

API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other. It provides a standardized interface for accessing and utilizing functionalities or data from external services, enabling seamless integration and interoperability between software components.



#### 7.1.1 Running API 

To run API : 

```
cd song_recommender
python3 src/api.py
```
check localhost:8000/docs for documentation.



#### 7.1.2 Request Format 

```
curl -X 'POST' \
  'http://localhost:8080/recommend/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "userid": 2,
  "item_to_recommend": 10
}'


```


#### 7.1.3 Response Format 
```
{
  "recommended_artist": [
    "Michael Jackson",
    "Roxette",
    "a-ha",
    "Lily Allen",
    "Annie Lennox",
    "Björk",
    "Rammstein",
    "Elvis Presley",
    "Norah Jones",
    "Vangelis"
  ]
}


```

## 8. Experiment with your own.
