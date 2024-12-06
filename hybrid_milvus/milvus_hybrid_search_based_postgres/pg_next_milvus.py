from pymilvus import *
import psycopg2
import getopt
import sys
from datetime import datetime
import numpy as np
from pymilvus import *
import concurrent.futures


COMPLETE_CSV_FILE = '/tmp/temp_full.csv'

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
MILVUS_collection = 'pg_milvus_collection'
VEC_DIM = 128

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

def load_fvecs_data_trial(fname,query_location):
    #begin_num = base_len * idx
    # print(fname, ": ", begin_num )
    x = np.memmap(fname, dtype='float32', mode='r')
    d = x[:4].view('int32')[0]
    #print(f"The value of the dimension is: {d}")
    #print(f"query location is: {query_location}")
    query_location = int(query_location)
    data =  x.reshape(-1, d + 1)[query_location:(query_location+1), 1:]   
    #data = (data + 0.5) / 255
    # data = normaliz_data(data)
    data = data.tolist()
    if(len(data) == 0):
        print(f"{query_location} data is zero")
        sys.exit()
    return data[0]

def copy_data_to_pg_using_psql():
    #fname = 'temp_full.csv'
    #csv_path = os.path.join(os.getcwd(), fname)
    #csv_path = '/tmp/temp.csv'
    csv_path = '/tmp/postgres_table.csv'

    # Construct the psql command
    psql_command = f"sudo -u {PG_USER} psql -U {PG_USER} -d {PG_DATABASE} -c \"\\copy {PG_TABLE_NAME} from '{csv_path}' with CSV delimiter '|'\""

    try:
        # Use subprocess to execute the psql command
        subprocess.run(psql_command, shell=True, check=True)
        print("Data copied successfully using psql!")
    except subprocess.CalledProcessError as e:
        print("Failed to copy data using psql:", str(e))

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

def main():
    milvus_client = MilvusClient(host=SERVER_ADDR, port=SERVER_PORT)
  
    conn = connect_postgres_server()
    cur = conn.cursor()
    #create_pg_table(conn,cur)

    #connections.connect(alias="default", host=SERVER_ADDR, port=SERVER_PORT)
    #milvus_col = Collection(MILVUS_collection)

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

    vectors  = []
    vector_ids = []
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

        elif opt_name in ("-q", "--query"):
            print("querying the values:")

            if sex_flag:
                if time_flag:

                    if glasses_flag:
                        print("Glasses flag")
                       
                    else:
                        pg_query_res = search_in_pg_1(conn,cur, sex,time_insert)
                        ids = [pg_tuple[0] for pg_tuple in pg_query_res]
                        print(f"Len of results is: {len(ids)}")
                        #fetch those vectors with row ids returned from the postgres
                        create_milvus_related_tables_and_index(MILVUS_collection, milvus_client)
                        parallel_load_and_insert(BASE_VECTORS_PATH, MILVUS_collection, ids)
                        print("Inserted successfully")

                        # for id in ids:
                        #     vector=load_fvecs_data_trial(fname,id)
                        #     vectors.append(vector)
                        #     vector_ids.append(id)

                        # mutation_result = milvus_col.insert([vectors_ids, vectors])



if __name__ == '__main__':
    main()