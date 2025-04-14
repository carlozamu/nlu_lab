#!/usr/bin/env python3

import os
import csv


def find_ideal_epoch(csv_path, loss_thresh=0.005, ppl_thresh=0.005):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prev_loss, prev_ppl = None, None

        for row in reader:
            epoch = int(row["Epoch"])
            dev_loss = float(row["Dev Loss"])
            dev_ppl = float(row["Dev PPL"])

            if prev_loss is not None and prev_ppl is not None:
                loss_delta = abs(dev_loss - prev_loss) / prev_loss
                ppl_delta = abs(dev_ppl - prev_ppl) / prev_ppl

                if loss_delta < loss_thresh and ppl_delta < ppl_thresh:
                    return {
                        "path": csv_path,
                        "epoch": epoch - 1,
                        "dev_loss": prev_loss,
                        "dev_ppl": prev_ppl,
                    }
            prev_loss, prev_ppl = dev_loss, dev_ppl

    return None


def main():
    root_dir = "."
    best_result = None

    for root, dirs, files in os.walk(root_dir):
        if "training_summary.csv" in files:
            csv_path = os.path.join(root, "training_summary.csv")
            result = find_ideal_epoch(csv_path)

            if result:
                if best_result is None or result["dev_ppl"] < best_result["dev_ppl"]:
                    best_result = result

    if best_result:
        print("Miglior configurazione trovata:")
        print(f"  - File:       {best_result['path']}")
        print(f"  - Epoca:      {best_result['epoch']}")
        print(f"  - Dev Loss:   {best_result['dev_loss']:.4f}")
        print(f"  - Dev PPL:    {best_result['dev_ppl']:.2f}")
    else:
        print("Nessun compromesso ideale trovato in nessun file.")


if __name__ == "__main__":
    main()
