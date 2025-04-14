#!/usr/bin/env python3

import os
import csv

def main():
    # Cartella da cui partire. Qui: . (cartella corrente)
    root_dir = '.'
    
    min_ppl = float('inf')
    min_path = None
    min_epoch = None

    for root, dirs, files in os.walk(root_dir):
        if 'training_summary.csv' in files:
            csv_path = os.path.join(root, 'training_summary.csv')
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Separa con virgola (CSV standard). Se i file sono tab-delimited, cambia con delimiter='\t'
                reader = csv.DictReader(f, delimiter=',')
                
                for row in reader:
                    dev_ppl = float(row['Dev PPL'])
                    epoch = row['Epoch']
                    
                    if dev_ppl < min_ppl:
                        min_ppl = dev_ppl
                        min_path = csv_path
                        min_epoch = epoch
    
    if min_path is not None:
        print(f"Il PPL minimo è {min_ppl}")
        print(f"  - Trovato nel file: {min_path}")
        print(f"  - All’epoca:        {min_epoch}")
    else:
        print("Nessun file training_summary.csv trovato.")

if __name__ == '__main__':
    main()
