##########################################################
####################### Libraries ########################
##########################################################

import uuid
import time
import random
import datetime
from influxdb_client import InfluxDBClient, Point, WriteOptions


##########################################################
###################### Parameters ########################
##########################################################

url="https://localhost:8086"
token = "<my-token>"
org = "<my-org>"
bucket = "<my-bucket>"
data_rate = 123
datatype = "Tempearture"
number_devices = 5

##########################################################
####################### Functions ########################
##########################################################

def data_generator_loop(input_client, data_rate:int) -> None:
    
    while True:
        # Iterate through number of device
        measurements = []
        for device_num in range(0, number_devices):
            tags = device_num
            tags["sensor_id"] = device_num
            measurements.append({
                "measurement": "temperature",
                "time": datetime.datetime.utcnow().isoformat() + 'Z',
                "tags": tags,
                "fields" : {
                    "value": random.randint(0,45)
                }
            })
            
            
        # Sleep
        
        break


##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    
    # Establish Client
    client = InfluxDBClient(
        url=url,
        token=token,
        org=org
    )

    # Entire Generator Loop
    data_generator_loop(
        input_client=client,
        
    )
    