import pandas as pd

# read a the txt file
with open('./Collection/combined_documents.txt', 'r') as f:
    text_data = f.read()

# Splitting the text into documents
documents = [doc.strip() for doc in text_data.split('********************************************\n') if doc.strip()]

# Extracting data and creating a DataFrame
data = {'doc_num': [], 'title': [], 'text': []}
seen_titles = set()
seen_doc = set()
for i, doc in enumerate(documents, start=0):
    lines = doc.split('\n', 2)
    title = lines[1].strip()
    text = lines[2].strip()
    num = int(''.join(filter(str.isdigit, str(lines[0].strip())))) # Extract numeric part
    data['doc_num'].append(num)
    data['title'].append(title)
    data['text'].append(text)

# Creating a DataFrame
df = pd.DataFrame(data)

df.to_csv('./Collection/documents.csv', index=False)

print(f"Successfully saved {len(df)} documents to './Collection/documents.csv'.")