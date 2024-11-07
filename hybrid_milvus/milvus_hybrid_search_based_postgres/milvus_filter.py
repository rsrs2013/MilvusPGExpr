# import face_recognition
import os
import time
from datetime import datetime
from pymilvus import *
import psycopg2
import numpy as np
import random
import subprocess
#from faker import Faker
import pandas as pd



MILVUS_collection = 'milvus_hybrid_query'
PG_TABLE_NAME = 'mixe_query'

#FILE_PATH = 'bigann_base.bvecs'
FILE_PATH = '/home/iitd/milvus/hybrid_milvus/sift/sift_base.fvecs'

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

#csv_file = "/tmp/temp_full.csv"
csv_file = "/tmp/postgres_table.csv"

# milvus = Milvus()

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

def create_milvus_collection(milvus):
    # Check if the collection already exists
    if milvus.has_collection(MILVUS_collection):
        print("Dropping the collection")
        print(MILVUS_collection)
        milvus.drop_collection(MILVUS_collection)

    # Define field for ID (Primary Key)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    
    # Define field for vector data (Float vector)
    vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)  # Adjust dimension as per your data
    
    # Define field for sex (Binary data - male or female)
    sex_field = FieldSchema(name="sex", dtype=DataType.VARCHAR, max_length=10)
    
    # Define field for glasses (Boolean data)
    glasses_field = FieldSchema(name="is_glasses", dtype=DataType.BOOL)
    
    # Define field for start_time (timestamp)
    time_field = FieldSchema(name="start_time", dtype=DataType.INT64)  # Store as Unix timestamp
    
    # Define the schema with all fields
    schema = CollectionSchema(
        fields=[id_field, vector_field, sex_field, glasses_field, time_field],
        #fields=[id_field, vector_field, sex_field, glasses_field],
        description="Example collection schema with vector and scalar fields"
    )

    collection = Collection(name=MILVUS_collection, schema=schema)

def build_milvus_index(milvus):
    print("Building Milvus indexes")
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
            index_type = "IVF_SQ8",
            metric_type = "L2",
            index_name="vector_index",
            params = {"nlist": 16384}
            )
    #index_param = {'nlist': 16384}
    status = milvus.create_index(MILVUS_collection,index_params)
    print(status)

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


def read_csv_to_arrays(csv_file):
    sex = []
    time = []
    is_glasses = []
    
    with open(csv_file, mode='r') as file:
        # Read the CSV file
        csv_reader = csv.reader(file, delimiter='|')
        
        for row in csv_reader:
            # Extract the fields from each row
            id_value = row[0]  # You can use this if needed
            sex_value = row[1]
            time_value = row[2].strip("'")  # Strip the single quotes around time
            is_glasses_value = row[3]

            # Append to respective arrays
            sex.append(sex_value)
            time.append(time_value)
            is_glasses.append(is_glasses_value)

    return sex, time, is_glasses

def read_csv_in_pandas_chunks(csv_file, chunk_size):
    # Use pandas to read the CSV file in chunks
    for chunk in pd.read_csv(csv_file, delimiter='|', chunksize=chunk_size, header=None):
        print("Reading the chunk")
        yield chunk

def start_timestamp(date_str):
    date_str = date_str.strip("'").split('.')[0]
    return int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())

def main():
    # connect_milvus_server()
    connections.connect(alias="default", host=SERVER_ADDR, port=SERVER_PORT)
    milvus = MilvusClient(host=SERVER_ADDR, port=SERVER_PORT)
    create_milvus_collection(milvus)
    build_milvus_index(milvus)
    count = 0
    chunk_size = BASE_LEN

    milvus_col = Collection(MILVUS_collection)
    milvus.load_collection(milvus_col.name)

    for chunk in read_csv_in_pandas_chunks(csv_file, chunk_size):
        ids = []
        sex = []
        start_time = []
        is_glasses = []
        print("Processing Chunk:")
        
        # Append each column to the respective array
        ids.extend(chunk.iloc[:, 0].tolist())  # First column (ID)
        sex.extend(chunk.iloc[:, 1].tolist())  # Second column (Sex)
        start_time.extend(chunk.iloc[:, 2].tolist())  # Third column (Get Time)
        is_glasses.extend(chunk.iloc[:, 3].tolist())  # Fourth column (Is Glasses)
        timestamps = [start_timestamp(date_str) for date_str in start_time]


        vectors_to_insert = load_fvecs_data_trial(FILE_PATH,BASE_LEN,count)
        print(f"Length of the vectors is: {len(vectors_to_insert)}")
        vectors_ids = [id for id in range(count*BASE_LEN,(count+1)*BASE_LEN)]
        time_start = time.time()    
        print(len(ids), len(sex), len(start_time), len(is_glasses))
        mutation_result = milvus_col.insert([vectors_ids, vectors_to_insert, sex, is_glasses, timestamps])

        
        #for i in range(10):
        #    print(str(ids[i]),sex[i],sep='\t')

        time_end = time.time()
        print(count, "insert milvue time: ", time_end-time_start)
        count = count + 1

        print("-" * 40)  # Separator for readability
    
    res = milvus.get(
    collection_name=MILVUS_collection,
    ids=[0, 1, 2]
    )
    print(res)

 
if __name__ == '__main__':
    main()
