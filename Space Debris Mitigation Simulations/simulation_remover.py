import csv

def remove_rows_with_value(input_csv: str, output_csv: str, value_to_remove: str) -> None:
    """
    Removes all rows from a CSV file that contain the given value.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the filtered CSV file.
        value_to_remove (str): The value to search for and remove rows containing it.
    """
    with open(input_csv, mode="r", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        filtered_rows = [row for row in reader if value_to_remove not in row]

    with open(output_csv, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(filtered_rows)

if __name__ == "__main__":
    remove_rows_with_value("output/data/lifetime_summary_2.csv", "output/data/lifetime_summary_2.csv", "6940")

