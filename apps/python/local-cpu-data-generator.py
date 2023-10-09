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
INFLUX_DB_URL = "http://localhost:8086/"
INFLUX_DB_USERNAME = "admin"
INFLUX_DB_PASSWORD = "admin123"
INFLUX_DB_ORG = "ai-playground"

# Generator Commands
MEASUREMENT_NAME = "census"
BUCKET_NAME = "ailen-observatory"

# Miscellanous
ailen_types = [
    "Water",
    "Fire",
    "Grass",
    "Electric",
    "Psychic",
    "Dark",
    "Bug",
    "Rock",
    "Ghost",
    "Fairy",
    "Ice",
    "Dragon",
    "Fighting",
    "Steel",
    "Ground",
    "Poison",
    "Flying",
    "Normal"
]


# SAMPLE
# python-module-3:
#     image: ai-stack-lite-run-1:latest
#     ports:
#       - 8003:5000/tcp
#     environment:
#       - RUN_TYPE=python
#       - RUN_SCRIPT_PATH=apps/python/cpu-data-generator.py
#       - INFLUX_DB_URL=http://influxdb:8086
#       - INFLUX_DB_USERNAME=admin
#       - INFLUX_DB_PASSWORD=admin123
#       - INFLUX_DB_ORD=ai-playground
#       - MEASUREMENT_NAME=census
#       - BUCKET_NAME=ailen-obersavtory

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
                ).field(
                    "sunlight", bool(random.getrandbits(1))
                ).field(
                    "lightray_size", round(random.uniform(0, 30), 2)
                ).field(
                    "ailen_type", random.choice(ailen_types)
                )
                
            write_api.write(bucket=BUCKET_NAME, org=INFLUX_DB_ORG, record=data_point)
        print(f"Inserted {count} records")
        time.sleep(1)
        count+=1