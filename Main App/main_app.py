import gradio as gr
import warnings
import utils
warnings.simplefilter(action='ignore', category=FutureWarning)

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="teal",
).set(
    body_background_fill='*neutral_50',
    body_text_color_subdued='*secondary_950',
    body_text_weight='300',
    background_fill_secondary='*neutral_100',
    background_fill_secondary_dark='*neutral_700',
    link_text_color='*primary_300',
    checkbox_background_color='*primary_700',
    checkbox_background_color_dark='*primary_400',
    checkbox_background_color_focus='*primary_50',
    checkbox_background_color_focus_dark='*primary_200',
    button_border_width='*checkbox_label_border_width',
    button_shadow='*button_shadow_hover',
    button_shadow_active='*button_shadow_hover',
    button_large_radius='*radius_xxl'
)
blocks = gr.Blocks(theme=theme,
                   css="""
        .header {
            text-align: center;
            padding: 0px;
            background-color: #f2f2f2;
        }
        .header img {
            max-width: 100%;
            max-height: 100%;
            margin-bottom: 0px;
        }
    """, title="Information Retrieval System")

with blocks as demo:
    gr.HTML("""<div class='header'><img src='http://localhost:8000/head.png' alt='Header Image'>     
            </div>""")    
    with gr.Row():
        tokenization = gr.Radio(["Split", "Regular Expression"], label="Toenization",value="Split")
        stemming = gr.Radio(["Lancaster","Poter"], label="Stemming",value="Lancaster")
        action = gr.Radio(["Search in the index", "Matching methods"], label="Action",value="Search in the index")
    with gr.Row(visible=True) as search_row:
        with gr.Column():
            with gr.Row():
                search_type = gr.Radio(["Documents per Term", "Terms per Documents"], label="Search Type",value="Documents per Term")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        search_bar = gr.Textbox(label="Search Bar")
                    with gr.Row():
                        search = gr.Button("Search")
            with gr.Row():
                search_result = gr.DataFrame()
    with gr.Row(visible=False) as matching_row:
        with gr.Column():
            with gr.Row():
                matching_type = gr.Radio(["Vector Space Model","Probabilistic Model","Boolean","Using embeddings"], label="Matching Type",value="Vector Space Model")
            with gr.Row(visible=True) as VSM:
                with gr.Column(scale=3):
                    vsm_model = gr.Dropdown(["Scalar Product","Cosine Similarity","Jacard mesure"], label="VSM Model",value="Scalar Product")
                with gr.Column(scale=1):
                    eval_match_vm = gr.Checkbox(label="Evaluation Test")
            with gr.Row(visible=False) as emb:
                with gr.Column(scale=3):
                    with gr.Row():
                        gr.Label("Please note that this method may take a long time to execute!",label="Warning")
                with gr.Column(scale=1):
                    eval_match_emb = gr.Checkbox(label="Evaluation Test")
            with gr.Row(visible=False) as probabilistic:
                with gr.Column(scale=3):
                    with gr.Row():
                        k = gr.Number(label="K",minimum=1.2,maximum=2.0,value=1.2)
                        b = gr.Number(label="B",minimum=0.5,maximum=0.75,value=0.5)
                with gr.Column(scale=1):
                    eval_match_pm = gr.Checkbox(label="Evaluation Test")
            with gr.Row(visible=True) as simple_match:
                with gr.Column():
                    with gr.Row():
                        matching_bar = gr.Textbox(label="Search Bar")
                    with gr.Row():
                        matching = gr.Button("Search")
            with gr.Row(visible=False) as eval_match_row:
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        with gr.Column(scale=4):
                                            matching_bar_eval = gr.Textbox(label="Search Bar",value=utils.get_query(1),lines=4,interactive=False)
                                        with gr.Column(scale=1):
                                            query_number = gr.Number(label="Query Number",step=1,minimum=1,maximum=35,precision=0,value=1)
                                    with gr.Row():
                                        matching_eval = gr.Button("Search")
                            with gr.Row(visible=False) as results_eval:
                                with gr.Column():
                                    precision = gr.Textbox(label="Precision",interactive=False)
                                    precision5 = gr.Textbox(label="Precision@5",interactive=False)
                                    precision10 = gr.Textbox(label="Precision@10",interactive=False)
                                    recall = gr.Textbox(label="Recall",interactive=False)
                                    f_score = gr.Textbox(label="F-Score",interactive=False)
                                with gr.Column():
                                    inter_plot = gr.Image(interactive=False)
            with gr.Row():
                matching_result = gr.DataFrame()
            

    def set_action(action):
        if action == "Search in the index":
            return {search_row: gr.Row(visible=True), matching_row: gr.Row(visible=False)}
        else:
            return {search_row: gr.Row(visible=False), matching_row: gr.Row(visible=True)}
    def search_index(search_bar, search_type, tokenization, stemming):
        if search_type == "Documents per Term":
            if search_bar == "":
                return gr.Warning("Please enter a term to search!")
            else:
                if " " in search_bar:
                    return gr.Warning("Please enter a single term!")
                else:
                    return utils.doc_per_term(search_bar,tokenization,stemming)
        elif search_type == "Terms per Documents":
            if search_bar == "":
                return gr.Warning("Please enter a document to search!")
            else:
                if not search_bar.isdigit():
                    return gr.Warning("Please enter a number!")
                else:
                    if int(search_bar) not in utils.get_availible_docs():
                        return gr.Warning("This Document isn't available! Please enter a valid document number! \n Available documents: 1-6004")
                    else:
                        return utils.term_per_doc(search_bar,tokenization,stemming)  
    def search_matching(matching_bar, matching_type, vsm_model,k,b, tokenization, stemming):
        if matching_bar == "":
            return gr.Warning("Please enter terms to search!")
        else:
            if matching_type == "Vector Space Model":
                return vector_space_model(matching_bar,vsm_model, tokenization, stemming)
            elif matching_type == "Probabilistic Model":
                return utils.dict_to_dataframe(utils.calculate_proba_model(matching_bar, tokenization, stemming,k,b))
            elif matching_type == "Boolean":
                if utils.est_syntaxe_requete_valide(matching_bar) == False:
                    return gr.Warning("Please enter a valid query!")
                else:
                    return utils.modele_booleen(matching_bar,tokenization,stemming)
            elif matching_type == "Using embeddings":
                return utils.embedding_based_cosin(matching_bar)
    def vector_space_model(matching_bar,vsm_model, tokenization, stemming):
        if vsm_model == "Scalar Product":
            return utils.dict_to_dataframe(utils.calculate_MPS(matching_bar,tokenization,stemming)) 
        elif vsm_model == "Cosine Similarity":
            return utils.dict_to_dataframe(utils.calculate_Cosine(matching_bar,tokenization,stemming))  
        elif vsm_model == "Jacard mesure":
            return utils.dict_to_dataframe(utils.calculate_Jaccard(matching_bar,tokenization,stemming))    
    def afficher_choix_matching(matching_type):
        if matching_type == "Vector Space Model":
            return {VSM: gr.Row(visible=True),probabilistic : gr.Row(visible=False),emb : gr.Row(visible=False)}
        elif matching_type == "Probabilistic Model":
            return {VSM: gr.Row(visible=False),probabilistic : gr.Row(visible=True),emb : gr.Row(visible=False)}  
        elif matching_type == "Boolean":
            return {VSM: gr.Row(visible=False),probabilistic : gr.Row(visible=False),emb : gr.Row(visible=False)}
        elif matching_type == "Using embeddings":
            return {VSM: gr.Row(visible=False),probabilistic : gr.Row(visible=False),emb : gr.Row(visible=True)}
    def eval_choice(eval_match):
        if eval_match == True:
            return {simple_match: gr.Row(visible=False), eval_match_row: gr.Row(visible=True)}
        else:
            return {simple_match: gr.Row(visible=True), eval_match_row: gr.Row(visible=False)}
    def evaluate_match(matching_bar_eval, matching_type, vsm_model,k,b, tokenization, stemming,query_number):
        if matching_type == "Using embeddings":
            df = utils.embedding_based_cosin_queries(query_number)
        else:
            df = search_matching(matching_bar_eval, matching_type, vsm_model,k,b, tokenization, stemming)
        unique_documents = set(df["Document"].unique())
        relevent_docs = utils.get_judgements_for_query(query_number)
        p = utils.precision(unique_documents,relevent_docs)
        p5 = utils.precision_at_k(df,relevent_docs,5)
        p10 = utils.precision_at_k(df,relevent_docs,10)
        r = utils.recall(unique_documents,relevent_docs)
        f = utils.f_score(p,r)
        return {results_eval : gr.Row(visible=True), matching_result: df, precision: p, precision5: p5, precision10: p10, recall: r, f_score: f, inter_plot: utils.plot_precision_recall_curve(df,relevent_docs)}
 
 
    action.select(inputs=[action],fn=set_action, outputs=[search_row,matching_row])
    search.click(inputs=[search_bar, search_type, tokenization, stemming],fn=search_index, outputs=[search_result])
    matching.click(inputs=[matching_bar, matching_type, vsm_model,k,b, tokenization, stemming],fn=search_matching, outputs=[matching_result])
    matching_type.select(inputs=[matching_type],fn=afficher_choix_matching, outputs=[VSM,probabilistic,emb])
    eval_match_vm.select(inputs=[eval_match_vm],fn=eval_choice, outputs=[simple_match,eval_match_row])
    eval_match_pm.select(inputs=[eval_match_pm],fn=eval_choice, outputs=[simple_match,eval_match_row])
    eval_match_emb.select(inputs=[eval_match_emb],fn=eval_choice, outputs=[simple_match,eval_match_row])
    query_number.change(inputs=[query_number],fn=utils.get_query, outputs=[matching_bar_eval])
    matching_eval.click(inputs=[matching_bar_eval, matching_type, vsm_model,k,b, tokenization, stemming,query_number],fn=evaluate_match, outputs=[results_eval,matching_result,precision,precision5,precision10,recall,f_score,inter_plot])

demo.launch()