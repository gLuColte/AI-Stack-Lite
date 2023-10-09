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
INFLUX_DB_URL = os.environ['INFLUX_DB_URL']
INFLUX_DB_USERNAME = os.environ['INFLUX_DB_USERNAME']
INFLUX_DB_PASSWORD = os.environ['INFLUX_DB_PASSWORD']
INFLUX_DB_ORG = os.environ['INFLUX_DB_ORG']

# Generator Commands
MEASUREMENT_NAME = os.environ['MEASUREMENT_NAME']
BUCKET_NAME = os.environ['BUCKET_NAME']

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
        username=INFLUX_DB_USERNAME, 
        password=INFLUX_DB_PASSWORD,
        org=INFLUX_DB_ORG
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    # p = influxdb_client.Point(measurement_name).tag("location", "Prague").tag("id", uuid.uuid4()).field("temperature", 25.3)
    count=1
    while True:
        for device_location_num in range(1,10):
            data_point = influxdb_client.Point(
                    MEASUREMENT_NAME
                ).tag(
                    "location", f"Sector-{device_location_num:02d}"
                ).field(
                    "temperature", round(random.uniform(0, 40), 2)
                ).field(
                    "human_count", random.randint(0, 20)
                ).field(
                    "ailen_count", random.randint(0, 5)
                )
            write_api.write(bucket=BUCKET_NAME, org=INFLUX_DB_ORG, record=data_point)
        print(f"Inserted {count} records")
        time.sleep(1)
        count+=1