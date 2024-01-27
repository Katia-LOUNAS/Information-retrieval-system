with open('./Collection/lisa/LISA.REL', 'r') as original_file:
    data = original_file.readlines()

with open('./Resultats/Judgements.txt', 'w') as output_file:
    current_query = None

    for line in data:
        if line.startswith('Query'):
            current_query = int(line.split()[1])
        elif 'Relevant Refs' in line:
            continue
        else:
            try:
                relevant_refs = map(int, line.replace('-1', '').split())
                for ref in relevant_refs:
                    output_file.write(f"{current_query}\t{ref}\n")
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue