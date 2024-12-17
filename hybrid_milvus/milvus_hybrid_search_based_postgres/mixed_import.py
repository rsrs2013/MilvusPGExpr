# import face_recognition
import os
import time
import csv
from pymilvus import *
import psycopg2
import numpy as np
import random
import subprocess
from faker import Faker
from postgres_manager import PostgresManager


fake = Faker()


MILVUS_collection = 'mixe_query'
PG_TABLE_NAME = 'mixe_query_prices'
PG_INDEX_NAME = 'mixe_query_prices_index'

#FILE_PATH = 'bigann_base.bvecs'
FILE_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_base.fvecs'
COMPLETE_CSV_FILE = '/tmp/temp_full.csv'

#VEC_NUM = 100000000
VEC_NUM = 1000000
BASE_LEN = 100000

VEC_DIM = 128

SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 19530

PG_HOST = "127.0.0.1"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DATABASE = "postgres"

# Number of rows in your database
num_rows = 1000000

# Define price range
price_min = 50
price_max = 1000

# Generate left-skewed random values using a beta distribution
alpha, beta = 5, 2  # Adjust alpha > beta for left skew
random_prices = np.random.beta(alpha, beta, num_rows)

# Scale the prices to your desired range
scaled_prices = price_min + (price_max - price_min) * random_prices

# Convert scaled prices to integers (round to the nearest integer)
scaled_prices = np.round(scaled_prices).astype(int)

# Optional: Check the data type and range
print(scaled_prices[:10])  # Print the first 10 prices as a sample
print(f"Data type: {scaled_prices.dtype}")
print(f"Min price: {scaled_prices.min()}, Max price: {scaled_prices.max()}")


# milvus = Milvus()

def load_bvecs_data(fname,base_len,idx):
    begin_num = base_len * idx
    # print(fname, ": ", begin_num )
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data =  x.reshape(-1, d + 4)[begin_num:(begin_num+base_len), 4:]   
    data = (data + 0.5) / 255
    # data = normaliz_data(data)
    data = data.tolist()
    return data

def load_fvecs_data_trial(fname,base_len,idx):
    begin_num = base_len * idx
    # print(fname, ": ", begin_num )
    x = np.memmap(fname, dtype='float32', mode='r')
    d = x[:4].view('int32')[0]
    data =  x.reshape(-1, d + 1)[begin_num:(begin_num+base_len), 1:]   
    #data = (data + 0.5) / 255
    # data = normaliz_data(data)
    data = data.tolist()
    return data

#def create_milvus_collection(milvus):
    #if not milvus.has_collection(MILVUS_collection):
        ##param = {
        ##    'collection_name': MILVUS_collection,
        ##    'dimension': VEC_DIM,
        ##    'metric_type':"L2"
        ##}
        ##milvus.create_collection(param)
        #
        ## 1. Create schema
        #schema = MilvusClient.create_schema(
            #auto_id=False,
            #enable_dynamic_field=False,
        #)  
#
        ## 2. Add fields to schema
        #schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
        #schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
#
        #milvus.create_collection(
                #collection_name=MILVUS_collection,
                #dimension=5,
                #metric_type="L2",
                #schema=schema
                #)

def create_milvus_collection(milvus):
    # Check if the collection already exists
    if milvus.has_collection(MILVUS_collection):
        print("Dropping the collection")
        print(MILVUS_collection)
        milvus.drop_collection(MILVUS_collection)

    collec = milvus.create_collection(
        collection_name=MILVUS_collection,
        dimension=VEC_DIM,
        metric_type="L2",
        auto_id=False
    )
    print(f"The list of collections are: {milvus.list_collections}")
    #print(f"The schema of the collection is {collec.schema}")
    print(milvus.describe_collection(MILVUS_collection))

def build_milvus_index(milvus):
    #list_of_indexes = milvus.list_indexes(collection_name=MILVUS_collection)
    status = milvus.release_collection(MILVUS_collection)
    print(f"Collection release status: {status}")
    milvus.drop_index(
        collection_name=MILVUS_collection, 
        index_name="vector"
    )
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
            field_name = "vector",
            index_type = "IVF_PQ",
            metric_type = "L2",
            index_name="vector_index",
            params = {"nlist": 128, "m":16}
            )
    #index_param = {'nlist': 16384}
    status = milvus.create_index(MILVUS_collection,index_params)
    print(status)

def connect_postgres_server():
    try: 
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
        print(f"Connected to the postgres server")
        return conn
    except Exception as e:
        print(f"Unable to connect to the database: {e}")
        return None  # Explicitly return None if the connection fails

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


def insert_data_to_pg(ids, vector, sex, get_time, is_glasses, conn, cur):
    sql = "INSERT INTO " + PG_TABLE_NAME + " VALUES(" + str(ids) + ", array" + str(vector) + ", '" + str(sex) + "', '" + str(get_time) + "', '" + str(is_glasses) + "');"
    print(sql)
    try:       
        # print(sql)
        cur.execute(sql)
        conn.commit()
        # print("insert success!")
    except:
        print("failed insert")


def copy_data_to_pg(conn, cur):
    fname = 'temp.csv'
    #csv_path = os.path.join(os.getcwd(),fname)
    csv_path = '/tmp/temp.csv'
    #sql = "\copy " + PG_TABLE_NAME + " from '" + csv_path + "' with CSV delimiter '|';"
    #print(sql)
    with open(csv_path, 'r') as f:
        try:
            next(f)
            cur.copy_from(f, PG_TABLE_NAME, sep='|')
            #cur.execute(sql)
            conn.commit()
            print("insert pg success MILVUS_collections ful!")
        except Exception as e:
            print(f"Failed with exception {e}")
            print("failed  copy!")

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

#def copy_data_to_pg(conn, cur):
#    fname = '/tmp/temp.csv'
#    csv_path = os.path.join(os.getcwd(),fname)
#    sql = "copy " + PG_TABLE_NAME + " from '" + csv_path + "' with CSV delimiter '|';"
#    # print(sql)
#    try:
#        cur.execute(sql)
#        conn.commit()
#        print("insert pg sucesMILVUS_collectionsful!")
#    except:
#        print("failed  copy!")


def build_pg_index(conn,cur):
    try:
        sql = "CREATE INDEX index_ids on " + PG_TABLE_NAME + "(ids);"
        cur.execute(sql)
        conn.commit()
        print("build index sucessful!")
    except Exception as e:
        print(f"failed build index {e}")

def build_pg_index_using_psql():
    psql_command = f"sudo -u {PG_USER} psql -U {PG_USER} -d {PG_DATABASE} -c \"CREATE INDEX index_ids on {PG_TABLE_NAME} (ids);\""
    print(f"The psql command is: {psql_command}")
    try:
        subprocess.run(psql_command, shell=True, check=True)
        print("build index sucessful!")
    except Exception as e:
        print(f"failed build index {e}")

# writes a line of string id, sex, time, glasses to a file
# this file is read by the postgres sql
def record_txt(ids, prices_to_insert):
    fname = '/tmp/temp.csv'
    f2 = open(COMPLETE_CSV_FILE, "a")
    with open(fname,'w+') as f:
        for i in range(len(ids)):
            #sex = random.choice(['female','male'])
            #get_time = fake.past_datetime(start_date="-120d", tzinfo=None)
            #is_glasses = random.choice(['True','False'])
            #line = str(ids[i]) + "|" + sex + "|'" + str(get_time) + "'|" + str(is_glasses) + "\n"
            price =  prices_to_insert[i]
            line = str(ids[i]) + "|" + str(price) + "\n"
            f.write(line)
            f2.write(line)

def read_fvecs(file_path, batch_len):
    with open(file_path, 'rb') as f:
        while True:
            vectors = []
            for _ in range(batch_len):
                # Read the dimensionality (first 4 bytes)
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break  # End of file
                dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
                # Read the vector (next 4 * dim bytes)
                vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
                vectors.append(vec)
            if not vectors:
                break  # No more vectors to read
            yield vectors

# Batch insert vectors into Milvus
def batch_insert_vectors(collection, file_path, batch_len):
    for batch_idx, vectors in enumerate(read_fvecs(file_path, batch_len)):
        data = [vectors]  # Wrap the list of vectors inside another list
        try:
            collection.insert(data)
            print(f"Inserted batch {batch_idx + 1} with {len(vectors)} vectors.")
        except Exception as e:
            print(f"Error inserting batch {batch_idx + 1}: {e}")
            break  # Stop if there is an error


#def read_fvecs(file_path):
#    with open(file_path, 'rb') as f:
#        # Read the dimensionality (first integer of each vector) as int32
#        while True:
#            # Read the first 4 bytes (which is the dimension of the vector)
#            dim_bytes = f.read(4)
#            if not dim_bytes:
#                break  # End of file
#            
#            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
#
#            # Now, read the vector of floats (4 * dim bytes)
#            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
#
#            yield vec


def main():
    # connect_milvus_server()
    milvus = MilvusClient(host=SERVER_ADDR, port=SERVER_PORT)
    create_milvus_collection(milvus)
    build_milvus_index(milvus)
    #conn = connect_postgres_server()
    pg_manager = PostgresManager(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE)
    # Connect to the PostgreSQL server
    pg_manager.connect()
    # Create the table
    TABLE_SCHEMA = "ids bigint, price int"
    pg_manager.create_table(TABLE_SCHEMA, PG_TABLE_NAME)
    #cur = conn.cursor()
    #create_pg_table(conn,cur)
    count = 0

    connections.connect(alias="default", host=SERVER_ADDR, port=SERVER_PORT)
    milvus_col = Collection(MILVUS_collection)

    try:
        # Attempt to open the file
        with open(COMPLETE_CSV_FILE,'r') as file:
            print("The file exists.")
            rm_command = "sudo rm " + COMPLETE_CSV_FILE 
            subprocess.run(rm_command, shell=True, check=True)
    except FileNotFoundError:
        print("The file does not exist.")

    # Inserting into Milvus
    while count < (VEC_NUM // BASE_LEN):
        #vectors = load_bvecs_data(FILE_PATH,BASE_LEN,count)
        vectors = load_fvecs_data_trial(FILE_PATH,BASE_LEN,count)
        print(f"Length of the vectors is: {len(vectors)}")
        #data = [vectors]
        # generate the vector ids for each of the row in the fvecs file
        vectors_ids = [id for id in range(count*BASE_LEN,(count+1)*BASE_LEN)]
        if len(vectors) != BASE_LEN:
            raise ValueError(f"Expected {BASE_LEN} vectors, but got {len(vectors)}.")
        else:
            print("Length of vectors is :", len(vectors))

        time_start = time.time()
        mutation_result = milvus_col.insert([vectors_ids, vectors])
        #status, ids = milvus.insert(collection_name=MILVUS_collection, records=vectors, ids=vectors_ids)
        #mutation_result = milvus_col.insert(data)

        time_end = time.time()
        print(count, "insert milvue time: ", time_end-time_start)
        # print(count)
        time_start = time.time()
        prices_to_insert = scaled_prices[count*BASE_LEN:(count+1)*BASE_LEN]
        record_txt(mutation_result.primary_keys, prices_to_insert)
        pg_manager.copy_data_from_csv('/tmp/temp.csv', PG_TABLE_NAME)
        #copy_data_to_pg_using_psql()
        time_end = time.time()
        print(count, "insert pg time: ", time_end-time_start)

        count = count + 1

    print("Total count is: ",count)
    #copy_data_to_pg_using_psql()
    #build_pg_index_using_psql()
    pg_manager.build_pg_index_using_psql(PG_TABLE_NAME, PG_INDEX_NAME)


if __name__ == '__main__':
    main()
