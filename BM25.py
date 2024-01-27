def calculate_proba_model(search_bar, tokenization, stemming, K, B):
    query = normaliser_terms(search_bar, tokenization, stemming)
    path = path_definition(1, tokenization, stemming)

    document = pd.read_csv(path, delimiter=' ', header=None, names=['term', 'frequency', 'doc_number', 'weight'])
    N = len(document['doc_number'].unique()) 

    # Filter document for query terms
    df = document.loc[document["term"].isin(query)] 
    
    # Calculate the number of documents containing each term
    nb_doc = df.groupby('term')['doc_number'].count().to_frame() 
    nb_doc.reset_index(inplace=True) 
    nb_doc.rename(columns = {'doc_number':'Nombre document contenu'}, inplace = True) # Renommage de la colonne
        
    # Merge with the original document dataframe
    df = df.merge(nb_doc, on="term") 

    # Calculate the number of terms per document
    nb_termes_doc = document.groupby('doc_number')['frequency'].sum().to_frame() 
    nb_termes_doc.reset_index(inplace=True) 
    nb_termes_doc.rename(columns = {'frequency':'dl'}, inplace = True)
        
    # Merge with the original document dataframe
    df = df.merge(nb_termes_doc, on="doc_number")
        
    # Calculate average document length
    df["avdl"]= nb_termes_doc["dl"].mean() 
        
    # Add avdl to the dataframe
    rsv_list = [] 

    for d in df["doc_number"].unique(): 
        temp = df.loc[df["doc_number"] == d, ["frequency", "Nombre document contenu", "dl", "avdl"]] 
        ni = temp["Nombre document contenu"] 
        rsv = np.multiply(np.divide(temp["frequency"], np.add(np.multiply(K,np.add(np.subtract(1, B), np.multiply(B, np.divide(temp["dl"], temp["avdl"])))), 
                                                    temp["frequency"])), np.log10(np.divide(np.add(np.subtract(N, ni), 0.5), np.add(ni, 0.5))))
            
        rsv_list.append([d, np.sum(rsv)]) 

    df = pd.DataFrame(rsv_list, columns = ["doc_number","RSV"]) 
    df = df.sort_values(by=["RSV"]) 

    return df.reindex(index=df.index[::-1])