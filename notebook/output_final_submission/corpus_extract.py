import csv

input_file = 'submission_final.csv'

# Đọc và xử lý dữ liệu
rows = []
with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    
    for row in reader:
        # Lấy corpus_id đầu tiên
        corpus_ids = row['corpus_ids'].split()
        if corpus_ids:
            row['corpus_ids'] = corpus_ids[0]
        rows.append(row)

# Ghi lại file
with open(input_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Đã cập nhật {input_file}")