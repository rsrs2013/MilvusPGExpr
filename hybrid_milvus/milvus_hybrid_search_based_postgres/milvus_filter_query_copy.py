import sys, getopt
import concurrent.futures
import time
from datetime import datetime
from pymilvus import *
import psycopg2
import numpy as np
import random
import subprocess
import pandas as pd
import struct
import os

MILVUS_collection = 'milvus_hybrid_query_with_price'

#FILE_PATH = 'bigann_base.bvecs'
FILE_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_base.fvecs'
QUERY_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_query.fvecs'
GROUND_TRUTH_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_groundtruth.ivecs'
GROUND_TRUTH_DIR = "output_ivecs_results"


VEC_NUM = 1000000
BASE_LEN = 100000
VEC_DIM = 128
TOP_K = 10

SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 19530
MILVUS_ALONE_RESULTS_FILE = "milvus_alone_res.csv"
TOTAL_QUERY_VECS = 5000
csv_file = "/tmp/temp_full.csv"

sex_flag = False
time_flag = False
glasses_flag = False
price_flag = False
filter_expr = ""

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def convert_ts_to_str(timestamp):
    # Assume you have a timestamp (e.g., Unix timestamp)
    #timestamp = 1633094400  # Example timestamp

    # Convert the timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object to a string using strftime
    formatted_string = dt_object.strftime('%Y-%m-%d %H:%M:%S')  # Example format

    return formatted_string

def record_recall_result(TOP_K, recall, precision):
    fname = MILVUS_ALONE_RESULTS_FILE
    with open(fname,'a') as f:
        line = str(TOP_K) + "|" + f"{recall * 100:.2f}" + "|" + f"{precision * 100:.2f}" + "|"+ "\n"
        f.write(line)

def get_vector_at_location(fname,query_location):
    #begin_num = base_len * idx
    # print(fname, ": ", begin_num )
    x = np.memmap(fname, dtype='float32', mode='r')
    d = x[:4].view('int32')[0]
    print(f"The value of the dimension is: {d}")
    print(f"query location is: {query_location}")
    query_location = int(query_location)
    data =  x.reshape(-1, d + 1)[query_location:(query_location+1), 1:]   
    #data = (data + 0.5) / 255
    # data = normaliz_data(data)
    data = data.tolist()
    return data


#def query_milvus(milvus_col, query_location, filter_expr, output_fields):
def query_milvus(milvus_col, filter_expr, output_fields):
    # Load the query vectors
    

    #query_location =  0
    #query_vec =get_vector_at_location(FILE_PATH,query_location)
    qvecs = fvecs_read(QUERY_PATH)

    # Define search parameters
    search_params = {
        "metric_type": "L2",  # Use L2 distance for nearest neighbor search
        "params": {"nprobe": 10}  # Adjust nprobe as needed
    }
    

    #start_date = '2023-05-01 00:00:00'
    #end_date = '2025-04-30 00:00:00'

    # Convert the start_date and end_date to Unix timestamps

    # Create the query dynamically using the provided parameters
    #filter_expr = f"(sex == 'male') and (start_time >= {start_timestamp}) and (get_time <= {end_timestamp})"
    #filter_expr = f"(sex == 'male')"

    #print("Filter expr is: ", filter_expr)
    #print(f"Output fields are: {output_fields}")
    
    # Perform the search
    result = milvus_col.search(
        data=qvecs,          # Query vectors
        anns_field="vector",         # Field to search (your vector field name)
        param=search_params,         # Search parameters
        output_fields=output_fields, 
        limit=TOP_K,                    # Top 
        expr=filter_expr                    # Optional expression for filtering (e.g., using other fields)
    )
    
    output_ids = []
    for hits in result:
        # get the IDs of all returned hits
        output_ids.append(hits.ids)

    return output_ids

# def query_milvus_for_vectors_with_ids(ground_truth_item, filter_expr, milvus_col):
#     # Construct the query expression for this particular ground_truth item
#     query_expression = f"id in {ground_truth_item} and {filter_expr}"
    
#     # Specify the fields you want to retrieve
#     output_fields = ["id"]
    
#     # Perform the query on Milvus
#     query_result = milvus_col.query(expr=query_expression, output_fields=output_fields)
    
#     # Return the result
#     return query_result


def generate_ground_truth(milvus_col):

    ground_truth = ivecs_read(GROUND_TRUTH_PATH).tolist()
    filtered_ground_truth = []

    # Step 4: Iterate over the 2D ground truth list
    # for vector_ids in ground_truth:
    #     # Convert list of vector IDs to a string format for the query
    #     query_expression = f"id in {vector_ids} and {filter_expr}"
    #     #print(query_expression)
    
    #     # Step 5: Query Milvus for the current row of vector IDs with the filter expression
    #     # Only retrieve the vector IDs that match the query
    #     output_fields = ["id"]  # Adjust based on the fields you need
    #     query_result = milvus_col.query(expr=query_expression, output_fields=output_fields)
    
    #     # Step 6: Extract the matching vector IDs from the result and store them
    #     # Query result will be a list of dictionaries, where 'id' field contains the vector ID
    #     filtered_vector_ids = [res["id"] for res in query_result]
    #     print(filtered_vector_ids)
    
    #     # Append the filtered vector IDs to the 2D result list
    #     filtered_ground_truth.append(filtered_vector_ids)
    count = 0
    for vector_ids in ground_truth:
        print(len(vector_ids))
        print (vector_ids)
        print(count)
        count = count + 1
        print("-"*40)

    for i in range(len(ground_truth)):
        query_expression = f"id in {ground_truth[i]} and {filter_expr}"
         #print(query_expression)
             # Step 5: Query Milvus for the current row of vector IDs with the filter expression
        # Only retrieve the vector IDs that match the query
        output_fields = ["id"]  # Adjust based on the fields you need
        query_result = milvus_col.query(expr=query_expression, output_fields=output_fields)
        print(f"Count is: {i}")
        print(f"Query result is: {query_result}")

    print("Filtered ground vectors")
    print(len(filtered_ground_truth))
    print(filtered_ground_truth)
    
    return filtered_ground_truth

# def run_multithreaded_queries(ground_truth, filter_expr, milvus_col, TOP_K=10, max_workers=10):
#     results_dict = {}  # Dictionary to store results with 'id' as the key

#     time_start_1 = time.time()
#     # Create a thread pool with a maximum number of workers
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit queries asynchronously and store the future results
#         future_to_query = {
#             executor.submit(query_milvus_for_vectors_with_ids, ground_truth[i], filter_expr, milvus_col): i
#             for i in range(len(ground_truth))
#         }

#         # As results complete, process them
#         for future in concurrent.futures.as_completed(future_to_query):
#             i = future_to_query[future]
#             try:
#                 query_result = future.result()  # Get the result of the query
#                 ids = [item['id'] for item in query_result[:TOP_K]]  # Get 'id' from first TOP_K items
#                 results_dict[i] = ids  # Store the result in the dictionary with 'id' as the key
#                 #print(f"Count is: {i}")
#                 #print(f"Query result is: {query_result}")
#             except Exception as exc:
#                 print(f"Query {i} generated an exception: {exc}")
    
#     time_end_1 = time.time()
#     print("Time to get the grounded truth is: ", time_end_1 - time_start_1)
#     for i in range(0,5):
#         print(results_dict[i])
#         print(len(results_dict[i]))
#     print(len(results_dict))

#     return results_dict

def calculate_recall_helper(predicted_indices, ground_truth_query_id, TOP_K=10):
    print(f"Value of TOP_K in recall helper is: {TOP_K}")
    print(f"Length of predicted_indices for query {ground_truth_query_id} is: {len(predicted_indices)}")
    k_sorted_gt = get_ground_truth(ground_truth_query_id, TOP_K)
    set_predicted = set(predicted_indices)
    set_k_gt = set(k_sorted_gt)
    print(f"Milvus Predicted set for query-{ground_truth_query_id} is: {set_predicted}")
    print(f"Gt set for query-{ground_truth_query_id} is: {k_sorted_gt}")

    true_positives = len(set_predicted & set_k_gt)
    actual_positives = len(k_sorted_gt)
    false_positives = len(set_predicted - set_k_gt)
    false_negatives = len(set_k_gt - set_predicted)

    print(f"QueryId:{ground_truth_query_id} AP {actual_positives}, FP {false_positives}, FN {false_negatives}, TP {true_positives}")
    return ground_truth_query_id, true_positives, actual_positives, false_positives, false_negatives


def calculate_recall(milvus_results, TOP_K=10, max_workers=10):
    time_start_1 = time.time()
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

# def calculate_recall(milvus_results, ground_truth):
#     correct_matches = 0
#     total_queries = len(ground_truth)

#     for i, result in enumerate(milvus_results):
#         predicted_indices = [hit.id for hit in result]
#         ground_truth_indices = ground_truth[i]
#         #print(f"Ground truth indices are: {ground_truth_indices}")
        
#         # Count how many of the predicted indices are in the ground truth
#         #s1 = set(predicted_indices)
#         #print(f"Predicted indices set is: {s1}")
#         #s2 = set(ground_truth_indices)
#         #print(f"ground_truth indices set  is:{s2}")
#         correct_matches += len(set(predicted_indices) & set(ground_truth_indices))

#     total_ground_truth = total_queries * len(ground_truth[0])
#     recall = correct_matches / total_ground_truth
#     return recall

#recall = calculate_recall(results, ground_truth)
#print(f"Recall: {recall * 100:.2f}%")



def main(argv):
    connections.connect(alias="default", host=SERVER_ADDR, port=SERVER_PORT)
    milvus = MilvusClient(host=SERVER_ADDR, port=SERVER_PORT)
    milvus_col = Collection(MILVUS_collection)
    milvus.load_collection(milvus_col.name)
    #filter_expr = None
    output_fields = []
    global TOP_K
    global price_flag
    global filter_expr
    
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "n:s:t:g:p:vk:q",
            ["num=", "sex=", "time=", "glasses=", "min-price=", "max-price=", "vector=", "top_k=", "query="],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
        sys.exit(2)

    for opt_name, opt_value in opts:
        #if opt_name in ("-k", "--knn"):
        #    TOP_K = opt_value
        #    print(f"TOP_K Nearest Neighbor: {opt_value}")
        if opt_name in ("-n", "--num"):
            query_location = opt_value
            #query_vec = load_query_list(FILE_PATH,query_location)
            query_vec = get_vector_at_location(FILE_PATH,query_location)

        elif opt_name in ("-s", "--sex"):
            print("Setting the gender")
            global sex_flag
            sex = opt_value
            sex_flag = True

        elif opt_name in ("-t", "--time"):
            print("Setting the time")
            time_insert = []
            global time_flag
            temp = opt_value
            time_insert.append(temp[1:20])
            time_insert.append(temp[22:41])
            st = datetime.strptime(time_insert[0], '%Y-%m-%d %H:%M:%S')
            et = datetime.strptime(time_insert[1], '%Y-%m-%d %H:%M:%S')

            start_timestamp = int(st.timestamp())
            end_timestamp = int(et.timestamp())

            time_flag = True

        elif opt_name in ("-g", "--glasses"):
            global glasses_flag
            glasses = opt_value
            glasses_flag = True

        elif opt_name in ("--min-price"):
            #global price_flag
            price_flag = True
            min_price = opt_value

        elif opt_name in ("--max-price"):
            #global price_flag
            price_flag = True
            max_price = opt_value

        elif opt_name in ("-q", "--query"):
            print("querying the values:")

            if price_flag:
                output_fields = ['price']
                filter_expr = f"(price >= {min_price}) and (price <= {max_price})"
            
            print("Searching in Milvus")
            #query_milvus(milvus_col, query_vec, filter_expr, output_fields)
            
            #get the results by querying the milvus for all the query vectors
            expr_results = query_milvus(milvus_col, filter_expr, output_fields)
            
            # get the vectors which satisfy the given criteria of 
            #print("Generating ground truth")
            #generate_ground_truth(milvus_col)
            #ground_truth = ivecs_read(GROUND_TRUTH_PATH).tolist()
            #ngt_dict = run_multithreaded_queries(ground_truth,filter_expr,milvus_col)
            calculate_recall(expr_results, TOP_K)

            sys.exit(2)

        elif opt_name in ("-v", "--vector"):
            id = opt_value
            conn = connect_postgres_server()
            cur = conn.cursor()
            search_vecs_pg(conn,cur,id)
        elif opt_name in ("-k", "--top_k"):
            try:
                TOP_K = int(opt_value)  # Convert the value to an integer
                print(f"Setting the top TOP_K value to {TOP_K}")
            except ValueError:
                print("Error: TOP_K must be an integer")
                sys.exit(2)
        else:
            print("wrong parameter")
            sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
