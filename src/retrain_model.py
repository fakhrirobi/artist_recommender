import argparse
import joblib
import os 
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix,csr_matrix
from implicit.evaluation import train_test_split


INTERIM_PATH = 'data/interim'


def initialize_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment Name",
        required=True,
    )
    parser.add_argument(
        "--training_file_name",
        type=str,
        help="Where the training path exists ",
        required=True,
    )

    args = parser.parse_args()
    return args

def load_utility_matrix(user_mapping,artist_mapping,training_file_name,
                        experiment_name
                        ) : 
    user_artist_path = os.path.join(INTERIM_PATH,training_file_name)
    artist_user_df = pd.read_csv(user_artist_path)
    
    user_id_to_ordered_id = {}
    ordered_id_to_user_id = {}
    for idx,user_id in enumerate(artist_user_df['userID'].unique()) : 
        user_id_to_ordered_id[user_id] = idx+1
        ordered_id_to_user_id[idx+1] = user_id
        
    artist_id_to_ordered_id = {}
    ordered_id_to_artist_id = {}
    for idx,artist_id in enumerate(artist_user_df['artistID'].unique()) : 
        artist_id_to_ordered_id[artist_id] = idx+1
        ordered_id_to_artist_id[idx+1] = artist_id

    joblib.dump(user_id_to_ordered_id,f'../mapping/{experiment_name}_user_id_to_ordered_id.pkl')
    joblib.dump(ordered_id_to_user_id,f'../mapping/{experiment_name}_ordered_id_to_user_id.pkl')
    joblib.dump(artist_id_to_ordered_id,f'../mapping/{experiment_name}_artist_id_to_ordered_id.pkl')
    joblib.dump(ordered_id_to_artist_id,f'../mapping/{experiment_name}_ordered_id_to_artist_id.pkl')
    

    artist_user_df.userID = user_artist_df.userID.map(user_mapping)
    artist_user_df.artistID = user_artist_df.artistID.map(artist_mapping)
        
    row = artist_user_df.userID.values
    col = artist_user_df.artistID.values
    data = artist_user_df.weight.values


    implicit_utility = coo_matrix((data,(row,col)))
    implicit_utility = implicit_utility.tocsr()
    return implicit_utility


if __name__ == "__main__" : 
    args = initialize_argparse()

    
    utility_matrix = load_utility_matrix(training_file_name=args.training_file_name,
                                         experiment_name=args.experiment_name)
    
    #convert utility as coo first for splitting
    utility_matrix_coo = utility_matrix.tocoo()


    train_data,test_data = train_test_split(ratings=utility_matrix_coo,train_percentage=0.8)
    
    trained_model= AlternatingLeastSquares(factors=100,
                                    regularization=0.16799704422342204,
                                    alpha=0.5097051938957499)
    trained_model.fit(train_data)
    
    final_tuned_metrics = ranking_metrics_at_k(model=trained_model,
                                   train_user_items=train_data,
                                   test_user_items=test_data)
    print(final_tuned_metrics)
    #fit on all dataset 
    trained_model.fit(utility_matrix)
    
    
    joblib.dump(trained_model,f"../models/{experiment_name}_als_tuned_model.pkl")
    print('Retraining Process Done')