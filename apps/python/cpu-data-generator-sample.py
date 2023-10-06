##########################################################
####################### Libraries ########################
##########################################################

import os
import uuid
import time
import random
import datetime
import traceback

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
##########################################################
###################### Parameters ########################
##########################################################

# Influx DB
INFLUX_DB_URL = "http://192.168.1.194:8086/"
INFLUX_DB_USERNAME = "admin"
INFLUX_DB_PASSWORD = "admin123"
INFLUX_DB_ORG = "ai-playground"


# Generator Commands
measurement_name = "census"
bucket_name = "alien-observatory"
datetime_format = '%Y-%m-%dT%H:%M:%S.%fZ'

##########################################################
####################### Functions ########################
##########################################################


##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    # Initialize Influx DB
    client = influxdb_client.InfluxDBClient(
        url=INFLUX_DB_URL,
        username='admin', 
        password='admin123',
        org=INFLUX_DB_ORG
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    # p = influxdb_client.Point(measurement_name).tag("location", "Prague").tag("id", uuid.uuid4()).field("temperature", 25.3)
    count=1
    while True:
        p = influxdb_client.Point(measurement_name).tag("location", "Meblourne").field("temperature", round(random.uniform(0, 40), 2))
        write_api.write(bucket=bucket_name, org=INFLUX_DB_ORG, record=p)
        print(f"Insert Count = {count}")
        time.sleep(1)
        count+=1