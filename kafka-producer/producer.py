from confluent_kafka import Producer
import csv
import json
import sys 
import time
import datetime

emission_file = 'data.csv'

fcd_topic = 'minnhen-fcd'
emission_topic = 'minnhen-emission'

kafka_url =  'kafka:9092'

def receipt(err, msg):
    if err is not None:
        print('Error: {}', format(err))
    else:
        message = 'Produces message on topic {}:{}'.format(msg.topic(), msg.value().decode('utf-8'))
        print(message)

if __name__ == '__main__':

    producer = Producer({'bootstrap.servers': kafka_url})
    print('Kafka Producer has been initiated')

    with open(emission_file) as emission_file:
        data = csv.DictReader(emission_file, delimiter=';')
        time_offset = 0
        while True:
            for row in data:
                emission_info = {}
                emission_info['timestep_time'] = float(row['timestep_time']) + time_offset * 21460
                emission_info['vehicle_CO'] = float(row['vehicle_CO']) 
                emission_info['vehicle_CO2'] = float(row['vehicle_CO2'])
                emission_info['vehicle_HC'] = float(row['vehicle_HC'])
                emission_info['vehicle_NOx'] = float(row['vehicle_NOx']) 
                emission_info['vehicle_PMx'] = float(row['vehicle_PMx']) 
                emission_info['vehicle_angle'] = float(row['vehicle_angle']) 
                emission_info['vehicle_eclass'] = row['vehicle_eclass']
                emission_info['vehicle_electricity'] = float(row['vehicle_electricity']) 
                emission_info['vehicle_id'] = row['vehicle_id'] 
                emission_info['vehicle_lane'] = row['vehicle_lane']
                emission_info['vehicle_fuel'] = float(row['vehicle_fuel']) 
                emission_info['vehicle_noise'] = float(row['vehicle_noise']) 
                emission_info['vehicle_pos'] = float(row['vehicle_pos']) 
                emission_info['vehicle_route'] = row['vehicle_route']
                emission_info['vehicle_speed'] = float(row['vehicle_speed']) 
                emission_info['vehicle_type'] = row['vehicle_type']
                emission_info['vehicle_waiting'] = float(row['vehicle_waiting']) 
                emission_info['vehicle_x'] = float(row['vehicle_x']) 
                emission_info['vehicle_y'] = float(row['vehicle_y'])

                vehicle_info = {}
                vehicle_info['timestep_time'] = float(row['timestep_time'])
                vehicle_info['vehicle_angle'] = float(row['vehicle_angle']) 
                vehicle_info['vehicle_id'] = row['vehicle_id'] 
                vehicle_info['vehicle_lane'] = row['vehicle_lane']
                vehicle_info['vehicle_pos'] = float(row['vehicle_pos']) 
                vehicle_info['vehicle_speed'] = float(row['vehicle_speed']) 
                vehicle_info['vehicle_type'] = row['vehicle_type']
                vehicle_info['vehicle_x'] = float(row['vehicle_x']) 
                vehicle_info['vehicle_y'] = float(row['vehicle_y']) 

                producer.produce(emission_topic, key = 'minnhen', value = json.dumps(emission_info), callback = receipt)
                producer.produce(fcd_topic, key = 'minnhen', value = json.dumps(vehicle_info), callback = receipt)
                producer.flush()
                time.sleep(0.1)

            time_offset += 1
        
    print('Kafka message producer done')