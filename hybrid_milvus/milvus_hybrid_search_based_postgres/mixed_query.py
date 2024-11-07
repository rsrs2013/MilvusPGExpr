import sys, getopt
import os
import time
from pymilvus import *
import psycopg2
import numpy as np
import struct
import concurrent.futures



#BASE_VECTORS_PATH = 'bigann_query.bvecs'
BASE_VECTORS_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_base.fvecs'
QUERY_VECTORS_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_query.fvecs'
GROUND_TRUTH_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_groundtruth.ivecs'
# query_location = 0

MILVUS_collection = 'mixe_query'
PG_TABLE_NAME = 'mixe_query'


SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530


PG_HOST = "127.0.0.1"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DATABASE = "postgres"

TOP_K = 40
DISTANCE_THRESHOLD = 1

ground_truth_dir = "output_ivecs_results"

# milvus = Milvus()


sex_flag = False
time_flag = False
glasses_flag = False
filter_expr = ""

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST,port=PG_PORT,user=PG_USER,password=PG_PASSWORD,database=PG_DATABASE)
        print("connect the database!")
        return conn
    except:
        print ("unable to connect to the database")


def load_fvecs_data_trial(fname,query_location):
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

def query_milvus_for_vectors_with_ids(ground_truth_item, filter_expr, milvus_col):
    # Construct the query expression for this particular ground_truth item
    query_expression = f"id in {ground_truth_item} and {filter_expr}"
    
    # Specify the fields you want to retrieve
    output_fields = ["id"]
    
    # Perform the query on Milvus
    query_result = milvus_col.query(expr=query_expression, output_fields=output_fields)
    
    # Return the result
    return query_result

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

def run_multithreaded_queries(ground_truth, filter_expr, milvus_col, K=10, max_workers=10):
    results_dict = {}  # Dictionary to store results with 'id' as the key

    time_start_1 = time.time()
    # Create a thread pool with a maximum number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit queries asynchronously and store the future results
        future_to_query = {
            executor.submit(query_milvus_for_vectors_with_ids, ground_truth[i], filter_expr, milvus_col): i
            for i in range(len(ground_truth))
        }

        # As results complete, process them
        for future in concurrent.futures.as_completed(future_to_query):
            i = future_to_query[future]
            try:
                query_result = future.result()  # Get the result of the query
                ids = [item['id'] for item in query_result[:K]]  # Get 'id' from first K items
                results_dict[i] = ids  # Store the result in the dictionary with 'id' as the key
                #print(f"Count is: {i}")
                #print(f"Query result is: {query_result}")
            except Exception as exc:
                print(f"Query {i} generated an exception: {exc}")
    
    time_end_1 = time.time()
    print("Time to get the grounded truth is: ", time_end_1 - time_start_1)
    for i in range(0,5):
        print(results_dict[i])
        print(len(results_dict[i]))
    print(len(results_dict))

    return results_dict

# Function to read an .ivecs file and return its content
# Here we are reading the ivecs file for each query. Each file has the base vectors sorted
# according to the eculidean distance and which satisfy the sql filter given by the user while
# generating the ivec file.
def read_ivecs_file(ivecs_filename):
    ivecs_filename = os.path.join(ground_truth_dir, ivecs_filename)
    with open(ivecs_filename, 'rb') as f:
        # Read the query ID (first 4 bytes)
        query_id = struct.unpack('i', f.read(4))[0]
        
        # Read the length of the sorted indices (next 4 bytes)
        length = struct.unpack('i', f.read(4))[0]
        
        # Read the sorted indices (length * 4 bytes)
        sorted_indices = list(struct.unpack(f'{length}i', f.read(length * 4)))
        return sorted_indices
  

def get_ground_truth(query_id, K = 10):
    ivecs_filename = f"query_{query_id}.ivecs"
    sorted_indices = read_ivecs_file(ivecs_filename)
    k_sorted_indices = sorted_indices[0:K]
    return k_sorted_indices
    #set_k_sorted_indices = set(k_sorted_indices)
    #return set_k_sorted_indices
    #intersection_result = [idx for idx in set_sorted_indices if idx in ids_set]

def calculate_recall_helper(query_id, milvus_result):
    #print(f"Reading {query_id} th query")
    predicted_indices = milvus_result
    k_sorted_gt = get_ground_truth(query_id, TOP_K)
    
    # Ground truth and predicted sets
    set_predicted = set(predicted_indices)
    set_k_gt = set(k_sorted_gt)
    
    # Calculate metrics
    actual_positives = len(k_sorted_gt)
    false_positives = len(set_predicted - set_k_gt)
    false_negatives = len(set_k_gt - set_predicted)
    true_positives = len(set_predicted & set_k_gt)
    
    print(f"AP {actual_positives}, FP {false_positives}, FN {false_negatives}, TP {true_positives}")

    # Optional: return result if you need to use the result elsewhere
    return actual_positives, false_positives, false_negatives, true_positives    

def calculate_recall(milvus_results):
    total_true_positives = 0
    # actual positives is false negatives + true positives
    total_actual_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    print(f"Length of milvus results are:{len(milvus_results)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:  # You can adjust the number of workers
        futures = [executor.submit(calculate_recall_helper, i, result) for i, result in enumerate(milvus_results)]

        # Aggregate results from each completed future
        for future in  concurrent.futures.as_completed(futures):
            actual_positives, false_positives, false_negatives, true_positives = future.result()
            total_actual_positives += actual_positives
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

    # for i, result in enumerate(milvus_results):
    #    #predicted_indices = [hit.id for hit in result]
    #     print(f"Reading {i} th query")
    #     predicted_indices = result
    #     k_sorted_gt = get_ground_truth(i, TOP_K)
    #     total_actual_positives += len(k_sorted_gt)
    #     #ground_truth_indices = ground_truth[i]
    #     #print(f"Ground truth indices are: {ground_truth_indices}")
      
    #     # Count how many of the predicted indices are in the ground truth
    #     #s1 = set(predicted_indices)
    #     #print(f"Predicted indices set is: {s1}")
    #     #s2 = set(ground_truth_indices)
    #     #print(f"ground_truth indices set  is:{s2}")
    #     set_predicted = set(predicted_indices)
    #     set_k_gt = set(k_sorted_gt)
    #     actual_positives = len(k_sorted_gt)
    #     false_positives = len(set_predicted - set_k_gt)
    #     false_negatives = len(set_k_gt - set_predicted)
    #     true_positives += len(set_predicted & set_k_gt)

              
    recall = total_true_positives / total_actual_positives
    precision = total_true_positives / (total_true_positives + total_false_positives) 
    print(f"Value of K is: {TOP_K}")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"True_positives: {total_true_positives} False Positives: {total_false_positives} False Negatives: {total_false_negatives} ")
    return recall

#recall = calculate_recall(results, ground_truth)
#print(f"Recall: {recall * 100:.2f}%")

def load_collection(milvus_client):
    collection = Collection(MILVUS_collection)
    print(f"Collection name is:{collection.name}")

    

        # Ensure the collection is loaded into memory
    #if not collection.has_loaded():
    print(f"Loading collection: {collection.name}")
    milvus_client.load_collection(collection.name)
    

def query_milvus(milvus_client, output_fields=None):
    qvecs = fvecs_read(QUERY_VECTORS_PATH)

    # Define search parameters
    search_params = {
        "metric_type": "L2",  # Use L2 distance for nearest neighbor search
        "params": {"nprobe": 10}  # Adjust nprobe as needed
    }
    load_collection(milvus_client)
    collection = Collection(MILVUS_collection)

    # Perform the search
    print(f"The number of qvecs are: {len(qvecs)}")
    result = collection.search(
        data=qvecs,          # Query vectors
        anns_field="vector",         # Field to search (your vector field name)
        param=search_params,         # Search parameters
        output_fields=output_fields, 
        limit=TOP_K,                    # Top 
        expr=None                    # Optional expression for filtering (e.g., using other fields)
    )
    
    output_ids = []
    output_distance = []
    for hits in result:
        # get the IDs of all returned hits
        output_ids.append(hits.ids)
        output_distance.append(hits.distances)
    output_ids = output_ids[0:5000]
    output_distance = output_distance[0:5000] 
    return output_ids, output_distance

def search_in_milvus(vector, milvus):
    output_ids = []
    output_distance = []
    print("searching in milvus")
    # Load the collection if not done already
    collection = Collection(MILVUS_collection)
    print(f"Collection name is:{collection.name}")

    

        # Ensure the collection is loaded into memory
    #if not collection.has_loaded():
    print(f"Loading collection: {collection.name}")
    milvus.load_collection(collection.name)
    
    # Assume the query vector is a list of 128-dimensional vectors (e.g., a single vector)
    query_vectors = np.random.random((1, 128)).tolist()
    
    # Define search parameters
    search_params = {
        "metric_type": "L2",  # Use L2 distance for nearest neighbor search
        "params": {"nprobe": 10}  # Adjust nprobe as needed
    }
    
    # Perform the search
    results = collection.search(
        data=vector,          # Query vectors
        anns_field="vector",         # Field to search (your vector field name)
        param=search_params,         # Search parameters
        limit=TOP_K,                    # Top 10 results
        expr=None                    # Optional expression for filtering (e.g., using other fields)
    )

    print(f"Results are: {results}")
    
    # Print search results
    for result in results:
        if result is None:
            print("It is None")
            break;
        for hit in result:
            #print(f"ID: {hit.id}, Distance: {hit.distance}")
            output_ids.append(hit.id)
            output_distance.append(hit.distance)
            
   # for result in results:
   #     # print(result)
   #     for i in range(TOP_K):
   #         if result[i].distance < DISTANCE_THRESHOLD:
   #             output_ids.append(result[i].id)
   #             output_distance.append(result[i].distance)

    return  output_ids,output_distance

# def merge_rows_distance(rows,ids,distance):
#     new_results = []
#     if len(rows)>0:
#         for row in rows:
#             index_flag = ids.index(row[0])
#             temp = [row[0]] + list(row[1:5]) + [distance[index_flag]]
#             print(temp)
#             new_results.append(temp)
#         new_results = np.array(new_results)
#         sort_arg = np.argsort(new_results[:,4])
#         new_results = new_results[sort_arg].tolist()
#         print("\nids                      sex        time                        glasses  distance")
#         for new_result in new_results:
#             print( new_result[0], "\t", new_result[1], new_result[2], "\t", new_result[3], "\t", new_result[4])
#     else:
#         print("no result")


def merge_rows_distance(list_of_pg_rows,milvus_ids,distance):
    merged_milvus_results = []

    if len(list_of_pg_rows)>0:
        for i, pg_row in enumerate(list_of_pg_rows):
            if len(pg_row) == 0:
                continue
            new_results = []
            for pk_id in pg_row:
                index_flag = milvus_ids[i].index(pk_id)
                temp = [pk_id] +  [distance[i][index_flag]]
                new_results.append(temp)

            new_results = np.array(new_results)
        
            sort_arg = np.argsort(new_results[:,1])
            new_results = new_results[sort_arg].tolist()

            merged_milvus_results.append(new_results)
        return merged_milvus_results
    else:
        print("no result")

def search_in_pg_0(conn,cur,milvus_query_ids,result_distance,sex,time,glasses):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "' and get_time between '" + time[0] + "' and '" + time[1] + "' and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")

def search_in_pg_1(conn,cur,milvus_query_ids,result_distance,sex,time):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "' and get_time between '" + time[0] + "' and '" + time[1] + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        #print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_2(conn,cur,milvus_query_ids,result_distance,sex,glasses):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "' and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_3(conn,cur,milvus_query_ids,result_distance,sex):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_4(conn,cur,milvus_query_ids,result_distance,time,glasses):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and get_time between '" + time[0] + "' and '" + time[1] + "' and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_5(conn,cur,milvus_query_ids,result_distance,time):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    # print(time[0])
    # print(time[1])
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and get_time between '" + time[0] + "' and '" + time[1] + "';"
    # print(sql)
    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_6(conn,cur,milvus_query_ids,result_distance,glasses):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")

def search_in_pg_7(conn,cur,milvus_query_ids,result_distance):
    sql1 = str(milvus_query_ids[0])
    i = 1
    while i < len(milvus_query_ids):
        sql1 = sql1 + "," + str(milvus_query_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ");"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search success!")
        print(len(rows))
        j = 0
        for row in rows:
            print(row[0], " ", row[1], " ", row[2], " ", row[3], " ", result_distance[j])
            j = j + 1
    except:
        print("search faild!")


def search_vecs_pg(conn,cur,id):
    sql = "select vecs from " + PG_TABLE_NAME + " where ids = " + id + ";"
    try:
        cur.execute(sql)
        rows=cur.fetchall()
        print(rows)
    except:
        print("search faild!")


def main(argv):
    connections.connect(alias="default", host=SERVER_ADDR, port=SERVER_PORT)

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "n:s:t:g:v:q",
            ["num=", "sex=", "time=", "glasses=", "query","vector="],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-n", "--num"):
            query_location = opt_value
            #query_vec = load_query_list(BASE_VECTORS_PATH,query_location)
            query_vec = load_fvecs_data_trial(BASE_VECTORS_PATH,query_location)

        elif opt_name in ("-s", "--sex"):
            global sex_flag
            sex = opt_value
            sex_flag = True

        elif opt_name in ("-t", "--time"):
            time_insert = []
            global time_flag
            temp = opt_value
            time_insert.append(temp[1:20])
            time_insert.append(temp[22:41])
            time_flag = True

        elif opt_name in ("-g", "--glasses"):
            global glasses_flag
            glasses = opt_value
            glasses_flag = True

        elif opt_name in ("-q", "--query"):
            milvus = MilvusClient(host=SERVER_ADDR, port=SERVER_PORT)
            milvus_col = Collection(MILVUS_collection)
            time_start_0 = time.time()
            print("Searching in Milvus")
            #milvus_query_ids, result_distance = search_in_milvus(query_vec,milvus)
            milvus_query_ids, result_distance = query_milvus(milvus)
            
            time_end_0 = time.time()            
            print("search in milvus cost time: ", time_end_0 - time_start_0)
            conn = connect_postgres_server()
            cur = conn.cursor()
            # print(sex_flag, glasses_flag,time_flag)

            

            
            if len(milvus_query_ids)>0:
                global filter_expr
                #ground_truth = ivecs_read(GROUND_TRUTH_PATH).tolist()
                postgres_rows = []
                
                if sex_flag:
                    if time_flag:
                        if glasses_flag:
                            # print(time[0])
                            # print(time[1])

                            # Get the k-nn from the milvus and filter them using postgres.
                            time_start_1 = time.time()
                            for row in milvus_query_ids:
                                rows = search_in_pg_0(conn,cur,row, result_distance, sex,time_insert,glasses)
                                postgres_rows.append(rows)
                            time_end_1 = time.time()
                            
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            #merge_rows_distance(rows,milvus_query_ids,result_distance)
                          

                        else:
                            time_start_1 = time.time()
                            list_of_empty_results = []

                            for i, row in enumerate(milvus_query_ids):
                                #print(row)
                                pg_query_res = search_in_pg_1(conn,cur,row, result_distance, sex,time_insert)
                                ids = [pg_tuple[0] for pg_tuple in pg_query_res]
                                if(len(ids) == 0):
                                    list_of_empty_results.append(i)
                                postgres_rows.append(ids)
                            
                            time_end_1 = time.time()
                          
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merged_results = merge_rows_distance(postgres_rows,milvus_query_ids,result_distance)
                          
                            #print(len(milvus_query_ids))
                            #print(f"{len(postgres_rows)}, {len(postgres_rows[0])}")
                            #print(f"{len(set(milvus_query_ids[0]) & set(postgres_rows[0]))}")

                            # for row in ground_truth:
                            #     #print (f" Number of intersections are: {len(set(row) & set(postgres_rows[key]))} ")
                            #     #print (f" Number of intersectons b/n milvus and postgres are: {len(set(milvus_query_ids[key]) & set(postgres_rows[key]))} ")
                            #     result = search_in_pg_1(conn,cur,row, result_distance, sex,time_insert)
                            #     ids = [item[0] for item in result]
                            #     ngt_dict[key] = ids
                            #     key = key + 1

                            # #print(f"NGT Dict is: {ngt_dict}")

                            print(f"The K value is {TOP_K}")
                            calculate_recall(postgres_rows)

                    else:
                        if glasses_flag:
                            time_start_1 = time.time()
                            rows = search_in_pg_2(conn,cur,milvus_query_ids, result_distance, sex, glasses)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,milvus_query_ids,result_distance)
                        else:
                            time_start_1 = time.time()
                            rows = search_in_pg_3(conn,cur,milvus_query_ids, result_distance,sex)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,milvus_query_ids,result_distance)
                else:
                    if time_flag:
                        if glasses_flag:
                            time_start_1 = time.time()
                            rows = search_in_pg_4(conn,cur,milvus_query_ids,result_distance,time_insert,glasses)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,milvus_query_ids,result_distance)
                        else:
                            time_start_1 = time.time()
                            rows = search_in_pg_5(conn,cur,milvus_query_ids,result_distance,time_insert)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,milvus_query_ids,result_distance)
                    else:
                        if glasses_flag:
                            time_start_1 = time.time()
                            rows = search_in_pg_6(conn,cur,milvus_query_ids,result_distance,glasses)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,milvus_query_ids,result_distance)
                        else:
                            time_start_1 = time.time()
                            search_in_pg_7(conn,cur,milvus_query_ids,result_distance)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                sys.exit(2)
            else:
                print("no vectors!")

        elif opt_name in ("-v", "--vector"):
            id = opt_value
            conn = connect_postgres_server()
            cur = conn.cursor()
            search_vecs_pg(conn,cur,id)
            sys.exit(2)

        else:
            print("wrong parameter")
            sys.exit(2)



if __name__ == "__main__":
    main(sys.argv[1:])
