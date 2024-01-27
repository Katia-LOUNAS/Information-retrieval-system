# Read data from the original file
with open('./Collection/lisa/LISA.QUE', 'r') as original_file:
    data = original_file.readlines()

with open('./Resultats/Queries.txt', 'w') as output_file:
    current_query = None
    in_query = False

    for line in data:
        line = line.rstrip()  

        if line.isdigit():
            current_query = line
            in_query = True
        elif line.endswith('#'):
            output_file.write(f"{line[:-1]}\n")
            in_query = False
        elif in_query:
            output_file.write(f"{line} ")
        elif not line:  
            continue
        else:
           
            output_file.write(f"{line}\n")
