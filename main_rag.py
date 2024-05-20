import os
import json
import streamlit as st
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from rag_chain import get_expression_chain
from langchain_core.tracers.context import collect_runs
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings

# Index Name
index_name = "earnings-call"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH"]['LANGSMITH_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI"]["OPENAI_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE"]['PINECONE_API_KEY']
os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGSMITH"]["LANGCHAIN_PROJECT"]

client = Client()
# embeddings = OpenAIEmbeddings()
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://rajrishabhrutuja.openai.azure.com/",
    azure_deployment="rishabh-text-embedding-ada-002",
    api_version="2024-02-01",
    api_key=os.environ.get("OPENAI_KEY")
)


# loading filenames to show in streamlit app
with open("mappings.json", 'r') as json_file:
    mappings = json.load(json_file)

@st.cache_resource(show_spinner=False)
def load_data():
    index = None
    try:
        index = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        print("Index:", index)
    except Exception as e:
        print(f"Could not load index: {e}")        
    return index

# Initializing index
if "index" not in st.session_state.keys():
	st.session_state.index = load_data()

st.set_page_config(
    page_title="Capturing User Feedback",
    page_icon="🦜️️🛠️",
)

st.subheader("🦜🛠️ Fractal Finance Bot")

# Metadata from user
with st.sidebar:
    year = st.selectbox("Select Year",list(mappings.keys()))
    quarter = st.selectbox("Select Quarter",list(mappings[year].keys()))
    file_name = st.selectbox("Select file", mappings[year][quarter])
	
metadata={"filename":file_name+".pdf","year":year,"quarter":quarter}

# Initializing session metadata
if "metadata" not in st.session_state.keys():
    st.session_state.metadata = metadata

# Initializing Query Engine
if "retriever" not in st.session_state.keys():
    # st.session_state.retriever = st.session_state.index.as_retriever(search_kwargs={"filter": st.session_state.metadata, "k": 4})
    st.session_state.chain = get_expression_chain(retriever=st.session_state.index)

# Updating filters and chat engine if metadata is updated and updating session metadata also
if st.session_state.metadata != metadata:
    st.session_state.metadata = metadata
    # st.session_state.retriever = st.session_state.index.as_retriever(search_kwargs={"filter": st.session_state.metadata, "k": 4})
    st.session_state.chain = get_expression_chain(retriever=st.session_state.index)

st.sidebar.markdown("## Feedback Scale")
feedback_option = (
    "thumbs" if st.sidebar.toggle(label="`Faces` ⇄ `Thumbs`", value=False) else "faces"
)

# Initialize session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

for msg in st.session_state.messages:
    print("MESSAGE:", msg)
    # avatar = "🦜" if msg.get("type") == "ai" else None
    with st.chat_message(msg.get("type")):
        st.markdown(msg.get("content"))


if prompt := st.chat_input(placeholder="Ask me a question!"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"type": "user", "content": prompt})
    with st.spinner(text="Thinking ..."):
        with st.chat_message("assistant"):
            context_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            # Define the basic input structure for the chains
            input_dict = {"query":prompt,
                          "k":4, 
                          "filter":{"filename":file_name+".pdf","year":year,"quarter":quarter}}
            with collect_runs() as cb:
                full_response = st.session_state.chain.invoke(input_dict)
                # print(full_response)
                st.session_state.messages.append({
                    "type": "ai",
                    "content": full_response.get("answer")
                })
                st.session_state.run_id = cb.traced_runs[0].id
            # context = [document.page_content for document in full_response.get("context_str")[0]]
            # meta_data = [document.metadata for document in full_response.get("context_str")[0]]
            # # context_placeholder.markdown("```" + "\n".join(context) + "```")
            # similarity_score = full_response.get("context_str")[0]
            # meta_data_string = ""
            context_count = 1
            context_string = ""
            for i in full_response.get("context_str"):
                meta_data_response = i[0].metadata
                # print(meta_data_response["filename"])
                # context_string += f"Context-{context_count}: <br>"+i[0].page_content+"\n\n" + "File References:"+ "<br>File Name:" + meta_data_response["filename"] + "<br>Page:" + str(meta_data_response["page"]) + "<br>Quarter:" + meta_data_response["quarter"] + "\nYear:" + meta_data_response["year"] + "<br>Similarity Score: "+str(round(i[1],3)*100)+"% <br><br><hr>"
                context_string+="""
**Context-{0}**:    
{1}

**File References**:      
- File Name: {2}
- Page: {3}
- Quarter: {4}
- Year: {5}

**Similarity Score**: {6}%    
***
                """.format(context_count,i[0].page_content, meta_data_response["filename"], meta_data_response["page"], meta_data_response["quarter"], meta_data_response["year"], str(round(i[1],3)*100))
                context_count+=1
            
            print(context_string)
            context_placeholder.markdown("### References:\n"+f"{context_string}")
            message_placeholder.markdown(full_response.get("answer"))

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{run_id}",
    )

    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option
            # and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string
            # and optional comment
            feedback_record = client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=feedback.get("text"),
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Invalid feedback score.")