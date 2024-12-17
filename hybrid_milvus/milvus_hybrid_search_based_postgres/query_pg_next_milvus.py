from pymilvus import *
import psycopg2
import getopt
import sys
from datetime import datetime
import numpy as np
from pymilvus import *
import concurrent.futures

import struct
import os


#COMPLETE_CSV_FILE = '/tmp/temp_full.csv'

PG_HOST = "127.0.0.1"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DATABASE = "postgres"
PG_TABLE_NAME = 'mixe_query'

SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 19530

BATCH_SIZE = 100000

sex_flag = False
time_flag = False
glasses_flag = False
filter_expr = ""

BASE_VECTORS_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_base.fvecs'
QUERY_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_query.fvecs'
GROUND_TRUTH_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_groundtruth.ivecs'
GROUND_TRUTH_DIR = "output_ivecs_results"
MILVUS_collection = 'pg_milvus_collection'
VEC_DIM = 128
TOP_K = 70

PG_NEXT_MILVUS_RESULTS_FILE = "pg_next_milvus.csv"
TOTAL_QUERY_VECS = 5000

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def query_milvus(milvus_col):
    # Load the query vectors
    qvecs = fvecs_read(QUERY_PATH)

    # Define search parameters
    search_params = {
        "metric_type": "L2",  # Use L2 distance for nearest neighbor search
        "params": {"nprobe": 10}  # Adjust nprobe as needed
    }
    
    # Perform the search
    result = milvus_col.search(
        data=qvecs,          # Query vectors
        anns_field="vector",         # Field to search (your vector field name)
        param=search_params,         # Search parameters
        limit=TOP_K,                    # Top 
    )
    
    output_ids = []
    for hits in result:
        # get the IDs of all returned hits
        output_ids.append(hits.ids)

    return output_ids

# def load_fvecs_data_trial(fname,query_location):
#     #begin_num = base_len * idx
#     # print(fname, ": ", begin_num )
#     x = np.memmap(fname, dtype='float32', mode='r')
#     d = x[:4].view('int32')[0]
#     #print(f"The value of the dimension is: {d}")
#     #print(f"query location is: {query_location}")
#     query_location = int(query_location)
#     data =  x.reshape(-1, d + 1)[query_location:(query_location+1), 1:]   
#     #data = (data + 0.5) / 255
#     # data = normaliz_data(data)
#     data = data.tolist()
#     if(len(data) == 0):
#         print(f"{query_location} data is zero")
#         sys.exit()
#     return data[0]

# def copy_data_to_pg_using_psql():
#     #fname = 'temp_full.csv'
#     #csv_path = os.path.join(os.getcwd(), fname)
#     #csv_path = '/tmp/temp.csv'
#     csv_path = '/tmp/postgres_table.csv'

#     # Construct the psql command
#     psql_command = f"sudo -u {PG_USER} psql -U {PG_USER} -d {PG_DATABASE} -c \"\\copy {PG_TABLE_NAME} from '{csv_path}' with CSV delimiter '|'\""

#     try:
#         # Use subprocess to execute the psql command
#         subprocess.run(psql_command, shell=True, check=True)
#         print("Data copied successfully using psql!")
#     except subprocess.CalledProcessError as e:
#         print("Failed to copy data using psql:", str(e))

def record_recall_result(TOP_K, recall, precision):
    fname = PG_NEXT_MILVUS_RESULTS_FILE
    with open(fname,'a') as f:
        line = str(TOP_K) + "|" + f"{recall * 100:.2f}" + "|" + f"{precision * 100:.2f}" + "|"+ "\n"
        f.write(line)

# Function to read an .ivecs file and return its content
# Here we are reading the ivecs file for each query. Each file has the base vectors sorted
# according to the eculidean distance and which satisfy the sql filter given by the user while
# generating the ivec file.
def read_ivecs_file(ivecs_filename):
    ivecs_filename = os.path.join(GROUND_TRUTH_DIR, ivecs_filename)
    with open(ivecs_filename, 'rb') as f:
        # Read the query ID (first 4 bytes)
        query_id = struct.unpack('i', f.read(4))[0]
        
        # Read the length of the sorted indices (next 4 bytes)
        length = struct.unpack('i', f.read(4))[0]
        
        # Read the sorted indices (length * 4 bytes)
        sorted_indices = list(struct.unpack(f'{length}i', f.read(length * 4)))
        return sorted_indices

def get_ground_truth(query_id, TOP_K = 10):
    ivecs_filename = f"query_{query_id}.ivecs"
    sorted_indices = read_ivecs_file(ivecs_filename)
    #print(f"Len of {query_id} is {len(sorted_indices)}")
    k_sorted_indices = sorted_indices[0:TOP_K]
    return k_sorted_indices

def calculate_recall_helper(predicted_indices, ground_truth_query_id, TOP_K=10):
    #print(f"Value of TOP_K in recall helper is: {TOP_K}")
    #print(f"Length of predicted_indices for query {ground_truth_query_id} is: {len(predicted_indices)}")
    k_sorted_gt = get_ground_truth(ground_truth_query_id, TOP_K)
    set_predicted = set(predicted_indices)
    set_k_gt = set(k_sorted_gt)
    #print(f"Milvus Predicted set for query-{ground_truth_query_id} is: {set_predicted}")
    #print(f"Gt set for query-{ground_truth_query_id} is: {k_sorted_gt}")

    true_positives = len(set_predicted & set_k_gt)
    actual_positives = len(k_sorted_gt)
    false_positives = len(set_predicted - set_k_gt)
    false_negatives = len(set_k_gt - set_predicted)

    #print(f"QueryId:{ground_truth_query_id} AP {actual_positives}, FP {false_positives}, FN {false_negatives}, TP {true_positives}")
    return ground_truth_query_id, true_positives, actual_positives, false_positives, false_negatives

def calculate_recall(milvus_results, TOP_K=10, max_workers=10):
    correct_matches = 0
    total_actual_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    recall_sum = 0
    precision_sum = 0
    # Create a thread pool with a maximum number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit queries asynchronously and store the future results
        future_to_query = {
            executor.submit(calculate_recall_helper, milvus_results[i],i, TOP_K): i
            for i in range(0,TOTAL_QUERY_VECS)
        }

        # As results complete, process them
        for future in concurrent.futures.as_completed(future_to_query):
            i = future_to_query[future]
            try:
                query_id, true_positives, actual_positives, false_positives, false_negatives = future.result()  # Get the result of the query
                local_recall = true_positives / actual_positives
                local_precision = 0
                if(true_positives + false_positives != 0):
                    local_precision = true_positives / (true_positives + false_positives)
                recall_sum += local_recall
                precision_sum += local_precision
                
                # correct_matches += intersection_res
                # total_actual_positives += actual_positives
                # total_false_positives += false_positives  
                # total_false_negatives += false_negatives              
            except Exception as exc:
                print(f"Query {i} generated an exception: {exc}")

    recall = recall_sum / TOTAL_QUERY_VECS
    precision = precision_sum / TOTAL_QUERY_VECS
    print(f"TOP_K value is: {TOP_K}")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    record_recall_result(TOP_K, recall, precision)
    #print(f"Total True Positives: {correct_matches} False Positives: {total_false_positives} False Negatives: {total_false_negatives} ")
    return recall
    
def main():
    milvus_client = MilvusClient(host=SERVER_ADDR, port=SERVER_PORT)
  
    #conn = connect_postgres_server()
    #cur = conn.cursor()
    #create_pg_table(conn,cur)

    connections.connect(alias="default", host=SERVER_ADDR, port=SERVER_PORT)
    milvus_col = Collection(MILVUS_collection)
    milvus_client.load_collection(milvus_col.name)

    global TOP_K
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "n:s:t:g:p:vk:q",
            ["num=", "sex=", "time=", "glasses=", "min-price=", "max-price=", "vector=", "top_k=", "query="],
        )
        # print(opts)
    except getopt.GetoptError:
        sys.exit(2)   
 
    for opt_name, opt_value in opts:
        if opt_name in ("-k", "--top_k"):
            try:
                TOP_K = int(opt_value)  # Convert the value to an integer
                print(f"Setting the top TOP_K value to {TOP_K}")
            except ValueError:
                print("Error: TOP_K must be an integer")
                sys.exit(2)

    output_ids = query_milvus(milvus_col)
    calculate_recall(output_ids,TOP_K)

 

if __name__ == '__main__':
    main()