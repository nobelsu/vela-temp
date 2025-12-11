
import csv

with open('vcbench_final_public.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['ipos'] or row['acquisitions']:
            print(f"Found row with exit info:")
            print(f"IPOs: {row['ipos']}")
            print(f"Acquisitions: {row['acquisitions']}")
            break
