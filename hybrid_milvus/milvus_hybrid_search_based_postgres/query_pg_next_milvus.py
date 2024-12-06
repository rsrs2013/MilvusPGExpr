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

def create_milvus_collection(milvus_client, collection_name):
    # Check if the collection already exists
    if milvus_client.has_collection(collection_name):
        print("Dropping the collection")
        print(collection_name)
        milvus_client.drop_collection(collection_name)

    collec = milvus_client.create_collection(
        collection_name=collection_name,
        dimension=VEC_DIM,
        metric_type="L2",
        auto_id=False
    )
    print(f"The list of collections are: {milvus_client.list_collections}")
    #print(f"The schema of the collection is {collec.schema}")
    print(milvus_client.describe_collection(collection_name))

def build_milvus_index(milvus, collection_name):
    #list_of_indexes = milvus.list_indexes(collection_name=MILVUS_collection)
    status = milvus.release_collection(collection_name)
    print(f"Collection release status: {status}")
    milvus.drop_index(
        collection_name=collection_name, 
        index_name="vector"
    )
    #index_params = MilvusClient.prepare_index_params()
    index_params = milvus.prepare_index_params()
    index_params.add_index(
            field_name = "vector",
            index_type = "IVF_SQ8",
            metric_type = "L2",
            index_name="vector_index",
            params = {"nlist": 16384}
            )
    #index_param = {'nlist': 16384}
    status = milvus.create_index(collection_name,index_params)
    print(status)

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

def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST,port=PG_PORT,user=PG_USER,password=PG_PASSWORD,database=PG_DATABASE)
        print("connect the database!")
        return conn
    except Exception as e:
        print ("unable to connect to the database", e)

def create_pg_table(conn,cur):
    try:
        print(f"Creating the postgres table")
        sql_drop = "DROP TABLE IF EXISTS "+ PG_TABLE_NAME;
        cur.execute(sql_drop);
        print(f"Dropped the table")
        sql = "CREATE TABLE " + PG_TABLE_NAME + " (ids bigint, sex char(10), get_time timestamp, is_glasses boolean);"
        cur.execute(sql)
        conn.commit()
        print("created postgres table!")
    except Exception as e:
        print(f"can't create postgres table: {e}")

def search_in_pg_1(conn,cur,sex,time):
    sql = "select * from " + PG_TABLE_NAME + " where sex='" + sex + "' and get_time between '" + time[0] + "' and '" + time[1] + "';"
    print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        #print(len(rows))
        return rows
    except:
        print("search failed!")

def parallel_load_and_insert(fname, collection_name, ids):
    vectors = []
    vector_ids = []
    connections.connect(alias="default", host=SERVER_ADDR, port=SERVER_PORT)
    milvus_col = Collection(collection_name)

    # Function to load data for a single ID
    def load_data(id):
        return id, load_fvecs_data_trial(fname, id)

    # Use ThreadPoolExecutor for parallel file reading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for all IDs
        futures = {executor.submit(load_data, id): id for id in ids}

        for future in futures:
            try:
                id, vector = future.result()  # Get the result
                vectors.append(vector)
                vector_ids.append(id)
            except Exception as e:
                print(f"Error loading data for ID {futures[future]}: {e}")

    #print(vectors[0:2])
    #print(vector_ids[0:2])
    # Insert data into Milvus
    VEC_NUM = len(vector_ids)
    count = 0
    # Function to perform batch insertion
    for i in range(0, VEC_NUM, BATCH_SIZE):
        # Slice the vector_ids and vectors arrays for the current batch
        batch_ids = vector_ids[i:i + BATCH_SIZE]
        batch_vectors = vectors[i:i + BATCH_SIZE]
    
        # Perform the insert operation
        mutation_result = milvus_col.insert([batch_ids, batch_vectors])
    
        # Optional: Print or log progress
        print(f"Inserted batch {i // BATCH_SIZE + 1} of {VEC_NUM // BATCH_SIZE + 1}")

def create_milvus_related_tables_and_index(collection_name, milvus_client):
    create_milvus_collection(milvus_client, collection_name)
    build_milvus_index(milvus_client, collection_name)

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
            "n:s:t:g:vk:q",
            ["num=", "sex=", "time=", "glasses=", "vector=", "top_k=", "query="],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
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