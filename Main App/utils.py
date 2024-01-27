import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import numpy as np
import tempfile
import os
import math
from nltk.stem import LancasterStemmer, PorterStemmer

nbrDocs = 6004

def path_definition(index,tokenization,stemming):
    if index == 0:
        if tokenization == "Split":
            if stemming == "Lancaster":
                path = "Resultats/DescripteurindexSplitLancester.txt"
            elif stemming == "Poter":
                path = "Resultats/DescripteurindexSplitPorter.txt"
        elif tokenization == "Regular Expression":
            if stemming == "Lancaster":
                path = "Resultats/DescripteurindexTokenLancester.txt"
            elif stemming == "Poter":
                path = "Resultats/DescripteurindexTokenPorter.txt"
    elif index == 1:
        if tokenization == "Split":
            if stemming == "Lancaster":
                path = "Resultats/InverseindexSplitLancester.txt"
            elif stemming == "Poter":
                path = "Resultats/InverseindexSplitPorter.txt"
        elif tokenization == "Regular Expression":
            if stemming == "Lancaster":
                path = "Resultats/InverseindexTokenLancester.txt"
            elif stemming == "Poter":
                path = "Resultats/InverseindexTokenPorter.txt"
    return path

def get_availible_docs():
    docs =[]
    with open("Resultats/DescripteurindexSplitLancester.txt") as file:
        for line in file:
                parts = line.split()
                if len(parts) == 4:
                    doc_num = int(parts[0])
                    docs.append(doc_num)
    return docs

def get_min_max():
    docs = get_availible_docs()
    return min(docs),max(docs)

def normalizeTerm(term, stemming):
    if stemming == "Lancaster":
        stemmer = LancasterStemmer()
    else:
        stemmer = PorterStemmer()
    return stemmer.stem(term.lower())

def normaliser_terms(terms,tokenization,stemming):
    if tokenization == "Split":
        terms = terms.split()
    else:
        ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*')
        terms = ExpReg.tokenize(terms)
    MotsVides = nltk.corpus.stopwords.words('english')
    terms = [term for term in terms if term.lower() not in MotsVides]
    normalised = []
    for term in terms:
        normalised.append(normalizeTerm(term, stemming))
    return normalised

def doc_per_term(search_bar,tokenization,stemming):
    index = 1
    filtered_data = []
    path = path_definition(index,tokenization,stemming)
    user_input = normalizeTerm(search_bar, stemming)
    with open(path, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 4 and parts[0] == user_input:
                filtered_data.append(parts)
    columns = ["Term", "Document", "Frequency", "Weight"]
    df = pd.DataFrame(filtered_data, columns=columns) 
    return df

def term_per_doc(search_bar,tokenization,stemming):
    index = 0
    filtered_data = []
    path = path_definition(index,tokenization,stemming)
    user_input = normalizeTerm(search_bar, stemming)
    with open(path, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 4 and parts[0] == user_input:
                filtered_data.append(parts)
    columns = ["Document", "Terms", "Frequency", "Weight"]
    df = pd.DataFrame(filtered_data, columns=columns) 
    return df

def calculate_MPS(search_bar,tokenization,stemming):
    normalised = normaliser_terms(search_bar,tokenization,stemming)
    path = path_definition(1,tokenization,stemming)
    min, max = get_min_max()
    result = [0.0] * (max - min + 1)
    with open(path, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 4 and parts[0] in normalised:
                result[int(parts[2]) - 1] = result[int(parts[2]) - 1] + float(parts[3])

    values = list(range(min, max + 1))
    my_dict = dict(zip(values, result))
    sorted_dict_desc = {key: value for key, value in my_dict.items() if value != 0.0}
    sorted_dict_desc = dict(sorted(sorted_dict_desc.items(), key=lambda item: item[1], reverse=True))

    return sorted_dict_desc

def calculate_Cosine(search_bar,tokenization,stemming):
    normalised = normaliser_terms(search_bar,tokenization,stemming)
    path = path_definition(1,tokenization,stemming)
    query_set = set(normalised)
    vocab_set = set()
    min, max = get_min_max()
    result = [0.0] * (max-min+1)
    v = 0.0
    w = [0.0] * (max-min+1)
    with open(path, "r") as file:
        for line in file:
            parts = line.split()
            w[int(parts[2])-1] += float(parts[3])**2
            vocab_set.add(parts[0])
            if len(parts) == 4 and parts[0] in normalised:
                result[int(parts[2])-1] = result[int(parts[2])-1]+float(parts[3])
    print(query_set)
    for term in query_set:
        if term in vocab_set:
            v += 1.0
    print(v)
    det = [v*w[i] for i in range(len(w))]
    det = [math.sqrt(det[i]) for i in range(len(det))]
    result = [result[i]/det[i] if det[i] != 0.0 else 0.0 for i in range(len(result)) ]
    values = list(range(min, max+1))
    my_dict = dict(zip(values, result))
    sorted_dict_desc = {key: value for key, value in my_dict.items() if value != 0.0}
    sorted_dict_desc = dict(sorted(sorted_dict_desc.items(), key=lambda item: item[1], reverse=True))                          
    return sorted_dict_desc

def calculate_Jaccard(search_bar,tokenization,stemming):
    normalised = normaliser_terms(search_bar,tokenization,stemming)
    path = path_definition(1,tokenization,stemming)
    min, max = get_min_max()
    query_set = set(normalised)
    vocab_set = set()
    v = 0.0
    result = [0.0] * (max-min+1)
    w = [0.0] * (max-min+1)
    with open(path, "r") as file:
        for line in file:
            parts = line.split()
            w[int(parts[2])-1] += float(parts[3])**2
            vocab_set.add(parts[0])
            if len(parts) == 4 and parts[0] in normalised:
                result[int(parts[2])-1] = result[int(parts[2])-1]+float(parts[3])
    for term in query_set:
        if term in vocab_set:
            v += 1.0
    det = [v+w[i]-result[i] for i in range(len(result))]
    result = [result[i]/det[i] if det[i] != 0.0 else 0.0 for i in range(len(result)) ]
    values = list(range(min, max+1))
    my_dict = dict(zip(values, result))
    sorted_dict_desc = {key: value for key, value in my_dict.items() if value != 0.0}
    sorted_dict_desc = dict(sorted(sorted_dict_desc.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict_desc

def calculate_proba_model(search_bar, tokenization, stemming, k, b):
    query = normaliser_terms(search_bar, tokenization, stemming)
    path = path_definition(1, tokenization, stemming)
    result = [0.0 for i in range(nbrDocs)]
    dl = [0.0 for i in range(nbrDocs)]
    avdl = 0.0
    freqi = [[0.0 for i in range(nbrDocs)] for j in range(len(query))]
    ni = [0.0 for i in range(len(query))]
    with open(path, "r") as file:
        for line in file:
            term,freq,num, weigth = line.split(sep=' ')
            dl[int(num)-1] += int(freq)
            if term in query:
                for index, value in enumerate(query):
                    if value == term:
                        freqi[index][int(num)-1] = int(freq)
                        ni[index] += 1
        for d in dl:
            avdl += d
        avdl = avdl/nbrDocs
    for i in range(nbrDocs):
        somme = 0.0
        for j in range(len(query)):
            somme += ((freqi[j][i])/(k*((1-b)+b*(dl[i]/avdl))+freqi[j][i])) * \
                (math.log10((nbrDocs-ni[j]+0.5)/(ni[j]+0.5)))
            result[i] = round(somme, 4)
    values = [str(i+1) for i in range(nbrDocs)]
    dict_result = dict(zip(values, result))
    sorted_dict = dict(
        sorted(dict_result.items(), key=lambda item: item[1], reverse=True))
    filtered_dict = {key: value for key,
                     value in sorted_dict.items() if value != 0}
    return filtered_dict

def doc_freq_length(df, d):
    return df[df['doc'] == d]['number'].sum()

def est_syntaxe_requete_valide(requete):
    pile = []
    opperandes = set(["AND", "OR", "NOT"])
    for terme in requete.split():
        terme = terme.upper()
        if terme in opperandes:
            if terme == "NOT":  
                if len(pile) > 0 and pile[-1] not in opperandes and pile[-1] != "(":
                    return False
                pile.append("NOT")  
            elif terme in ["AND", "OR"]:
                if len(pile) == 0 or pile[-1] in opperandes or pile[-1] == "(":
                    return False
                pile.append(terme)
        else:
            if pile and pile[-1] == "Terme":
                return False
            if pile and pile[-1] == "NOT":
                pile.pop() 
            pile.append("Terme")
    if len(pile) < 1 or pile[-1] in opperandes or pile[-1] == "(":
        return False
    if "NOT" in pile:
        not_index = pile.index("NOT")
        if not_index == 0 or pile[not_index - 1] in opperandes or pile[not_index - 1] == "(":
            return False
    return True

def read_inverted_index(file_path):
    inverted_index = {}
    with open(file_path, 'r') as file:
        for line in file:
            term, frequency, document, weight = line.split()
            inverted_index[(term, int(document))] = (int(frequency), int(document), float(weight))
    return inverted_index

def modele_booleen(query,tockenization,stemming):
    inverted_index = read_inverted_index(path_definition(1,tockenization,stemming))
    operators = {'and', 'or', 'not'} 
    def evaluate_term(term, document):
        return 1 if (term, document) in inverted_index else 0
    def evaluate_expression(expression, document):
        list_to_eval = []
        for token in expression:
            if token not in operators:
                list_to_eval.append(evaluate_term(token, document))
            else:
                list_to_eval.append(token)
        list_to_eval = [str(i) for i in list_to_eval]
        list_to_eval = ' '.join(list_to_eval)
        result = eval(list_to_eval)
        return result
              
    relevant_documents = []
    query = query.lower()
    if stemming == "Poter":
        Porter = nltk.PorterStemmer()
        query_tokens = []
        for token in query.split():
            if token not in operators:
                query_tokens.append(Porter.stem(token))
            else:
                query_tokens.append(token)
    else:
        Lancaster = nltk.LancasterStemmer()
        query_tokens = []
        for token in query.split():
            if token not in operators:
                query_tokens.append(Lancaster.stem(token))
            else:
                query_tokens.append(token)
    relevant_documents = []
    for document_id in set(doc for term, doc in inverted_index.keys()):
        result = evaluate_expression(query_tokens, document_id)
        if result:
            relevant_documents.append(document_id)
    df = pd.DataFrame({"RelevantDocs": relevant_documents})
    return df

def dict_to_dataframe(sorted_dict):
    df = pd.DataFrame(list(sorted_dict.items()), columns=["Document", "Relevance"])
    df["Relevance"] = df["Relevance"].map("{:.4f}".format)

    return df

def get_doc_queries(file_path):
    queries = []
    with open(file_path, 'r') as file:
        for line in file:
            queries.append(line.strip())
    return queries

def get_query(query_number):
    queries = get_doc_queries("Resultats/Queries.txt")
    return queries[query_number - 1]

def get_judgements():
    with open('Resultats/Judgements.txt', 'r') as f:
        judgements = [tuple(map(int, line.strip().split('\t'))) for line in f]
    return judgements

def get_judgements_for_query(query_number, judgements=get_judgements()):
        return {doc for q, doc in judgements if q == query_number}

def precision(selected_docs, relevant_docs):
        if len(selected_docs) == 0:
            return 0
        return len(selected_docs.intersection(relevant_docs)) / len(selected_docs)

def precision_at_k(selected_docs, relevant_docs, k):
    selected_docs_at_k = get_top_k_documents(selected_docs, k)
    num_relevant_in_selection = sum(1 for doc in selected_docs_at_k if doc in relevant_docs)
    precision_value = num_relevant_in_selection / k if k > 0 else 0.0
    return precision_value

def recall(selected_docs, relevant_docs):
    if len(relevant_docs) == 0:
        return 0
    return len(selected_docs.intersection(relevant_docs)) / len(relevant_docs)

def recall_at_k(selected_docs, relevant_docs, k):
    num_relevant_docs = len(relevant_docs)
    selected_docs_at_k = get_top_k_documents(selected_docs, k)
    num_relevant_in_selection = sum(1 for doc in selected_docs_at_k if doc in relevant_docs)
    recall_value = num_relevant_in_selection / num_relevant_docs if num_relevant_docs > 0 else 0.0
    return recall_value

def f_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def get_top_k_documents(df, k):
    top_k_documents = df.sort_values(by="Relevance", ascending=False).head(k)["Document"].tolist()
    return top_k_documents

def plot_precision_recall_curve(df, relevant_docs):
    precision_values = []
    recall_values = []
    for k in range(0, 11):
        precision_values.append(precision_at_k(df, relevant_docs, k))
        recall_values.append(recall_at_k(df, relevant_docs, k))
    interpolated_precision = []
    recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(0,11):
        rj = recall_levels[i]
        sup_prec = []
        for prec, recall in zip(precision_values, recall_values):
            if recall >= rj:
                sup_prec.append(prec)
            else:
                sup_prec.append(0.0)
        if len(sup_prec) != 0:
            max_precision = max(sup_prec)
            interpolated_precision.append(max_precision)
    plt.plot(recall_levels, interpolated_precision, label='Interpolated Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Interpolated Precision')
    plt.title('Interpolated Precision-Recall Curve')
    plt.legend()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file, format='png')
        temp_file_path = temp_file.name

    return temp_file_path

def embedding_based_cosin_queries(query_number):
    df = pd.read_csv("./Collection/documents.csv")
    loaded_embeddings = np.load("./document_embeddings.npy")
    with open("embeddings_dict.pkl", "rb") as f:
        loaded_embeddings_dict = pickle.load(f)
    embedding_to_retrieve = loaded_embeddings_dict.get(query_number)
    similarities = cosine_similarity(embedding_to_retrieve, loaded_embeddings)

    sorted_indices = similarities.argsort()[0][::-1]

    doc_numbers = []
    similarity_scores = []

    for index in sorted_indices:
        doc_number = df['doc_num'][index]
        similarity_score = similarities[0][index]
        doc_numbers.append(doc_number)
        similarity_scores.append(similarity_score)

    result_df = pd.DataFrame({'Document': doc_numbers, 'Relevance': similarity_scores})

    return result_df

def embedding_based_cosin(query):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    df = pd.read_csv("./Collection/documents.csv")
    loaded_embeddings = np.load("./document_embeddings.npy")
    query_tokens = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        query_outputs = model(**query_tokens)
    query_embedding = query_outputs.last_hidden_state.mean(dim=1).numpy()
    similarities = cosine_similarity(query_embedding, loaded_embeddings)
    sorted_indices = similarities.argsort()[0][::-1]

    doc_numbers = []
    similarity_scores = []

    for index in sorted_indices:
        doc_number = df['doc_num'][index]
        similarity_score = similarities[0][index]
        doc_numbers.append(doc_number)
        similarity_scores.append(similarity_score)

    result_df = pd.DataFrame({'Document': doc_numbers, 'Relevance': similarity_scores})

    return result_df
