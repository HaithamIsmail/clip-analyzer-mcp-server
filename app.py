import gradio as gr
import lancedb
import os
from dotenv import load_dotenv, find_dotenv
from utils import get_image_embedding, get_text_embedding
from video_pipeline import run_pipeline
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# --- Configuration ---
LANCEDB_URI = "output/lancedb" 
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_EMBEDDING_MODEL_NAME = "thenlper/gte-large"
CLIP_EMBEDDING_DIM = 512
HF_API_KEY = os.getenv("HF_API_KEY", "default_key")
CLIP_EMBEDDING_URL = os.getenv("MODAL_EMBEDDING_SERVER")

chat_model = ChatOpenAI(
    # model="Qwen/Qwen3-30B-A3B",
    model="Qwen/Qwen3-32B",
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

def get_hf_headers():
    """Get headers for Hugging Face API requests."""
    if not HF_API_KEY or "default_key" in HF_API_KEY:
        raise ValueError("HF_API_KEY environment variable not set or is a placeholder.")
    return HF_API_KEY

def search_clips(query_text):
    """Searches the LanceDB database for clips matching the query."""
    try:
        # Create embedding for the query using Hugging Face API
        query_vector = get_text_embedding(query_text, CLIP_EMBEDDING_URL)[0]
        
        # Connect to LanceDB
        db = lancedb.connect(LANCEDB_URI)
        table = db.open_table("video_clips")
        
        # Search for similar clips
        results = table.search(query_vector).limit(3).to_pandas()
        return results
        
    except FileNotFoundError:
        return f"Error: Database not found at {LANCEDB_URI}. Please ensure the video analysis server has processed some videos first."
    except Exception as e:
        return f"Error during search: {str(e)}"

def format_search_results(results_df):
    """Format search results for display."""
    if isinstance(results_df, str):  # Error message
        return results_df
    
    if results_df.empty:
        return "No clips found matching your query."
    
    response = "Here are the top results I found:\n\n"
    for idx, row in results_df.iterrows():
        response += f"**Clip {row.get('clip_id', 'N/A')} from {row.get('video_name', 'Unknown')}**\n"
        response += f"⏰ Time: {row.get('start_time', 'N/A')}s - {row.get('end_time', 'N/A')}s\n"
        
        # Handle summary safely
        summary = row.get('summary', 'No summary available')
        if isinstance(summary, str) and '---' in summary:
            summary = summary.split('---')[0].strip()
        
        response += f"📝 Summary: {summary}\n"
        
        # Add score if available
        if '_distance' in row:
            score = 1 - row['_distance']  # Convert distance to similarity score
            response += f"🎯 Relevance: {score:.2f}\n"
        
        response += "\n---\n\n"
    
    return response

@tool
def get_relevant_clips(query):
    """Retreive relevant clips from vector database

    Args:
        query: Text to use in vector search

    Returns :
        str: the search results formatted in a string
    """
    search_result = search_clips(query)
    formatted_search_result = format_search_results(search_result)

    return formatted_search_result

def chat_agent(message, history):
    messages = history or []
    messages.append({"role": "user", "content": message})

    llm_with_tool = chat_model.bind_tools(tools=[get_relevant_clips])
    tools = {"get_relevant_clips": get_relevant_clips}

    # The agent loop
    while True:
        # Pass the entire message history to the model
        ai_response = llm_with_tool.invoke(messages)
        
        # Append the assistant's response to the history
        messages.append(ai_response) # LangChain AIMessage objects work directly here

        # If there are no tool calls, the agent is done
        if not ai_response.tool_calls:
            break

        # If there are tool calls, process them
        for tool_call in ai_response.tool_calls:
            tool_output = tools[tool_call["name"]].invoke(tool_call)
            messages.append(tool_output)
    
    print(ai_response)
    # Extract content after </think> if it exists, otherwise use the content as is
    content = ai_response.content
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content

def check_database_status():
    """Check if the database exists and has data."""
    try:
        db = lancedb.connect(LANCEDB_URI)
        print(db.table_names())
        table = db.open_table("video_clips")
        count = len(table.to_pandas())
        return f"✅ Database connected. Found {count} video clips."
    except Exception as e:
        return f"❌ Database issue: {str(e)}"

def check_server_status():
    """Check if the MCP server is running."""
    try:
        import requests
        response = requests.get("http://localhost:7861/", timeout=5)
        return "✅ Video Analysis MCP Server is running on port 7861"
    except Exception:
        return "❌ Video Analysis MCP Server is not accessible on port 7861"

# Create the Gradio interface
with gr.Blocks(title="Video Search Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Video Search Agent")
    gr.Markdown("Search through your processed video clips using natural language queries.")
    
    # Status section
    with gr.Accordion("System Status", open=False):
        status_text = gr.Textbox(
            label="Status",
            value=f"{check_database_status()}\n{check_server_status()}",
            interactive=False,
            lines=3
        )
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(
            fn=lambda: f"{check_database_status()}\n{check_server_status()}",
            outputs=status_text
        )

    # Main chat interface
    gr.ChatInterface(
        fn=chat_agent,
        title="Chat with Video Search Agent",
        description="Ask me to find clips or analyze videos!",
        examples=[
            "find clips about cooking",
            "search clips about meetings",
            "clips about sports highlights",
            "What can you help me with?"
        ],
        type='messages'
    )
    
    # Video analyzer tool section
    with gr.Accordion("📹 Video Analyzer Tool", open=False):
        gr.Markdown("""
        **To analyze new videos:**
        1. Make sure the Video Analysis MCP Server is running on port 7860
        2. Upload your video file using the interface below
        3. The processed clips will be automatically added to the searchable database
        """)
        
        # Try to load the external tool
        try:
            # Note: gr.load() for external servers might not work as expected
            # You might need to implement a custom interface or use iframe
            gr.Markdown("🔗 **Video Analyzer Interface**")
            gr.Markdown("Open http://localhost:7860/ in a new tab to access the video analyzer.")
            
            # Alternative: Create a simple upload interface that calls the server
            with gr.Row():
                video_file = gr.File(
                    label="Upload Video",
                    file_types=[".mp4"],
                    type="binary"
                )
                analyze_btn = gr.Button("Analyze Video", variant="primary")
            
            analysis_output = gr.Textbox(
                label="Analysis Status",
                lines=3,
                interactive=False
            )
            
            def analyze_video_local(file_obj):
                import tempfile, shutil
                if not file_obj:
                    return "Please select a video file first."
                try:
                    # Save uploaded file to a temp file with the same name as the uploaded file
                    if hasattr(file_obj, 'name'):
                        original_filename = os.path.basename(file_obj.name)
                    else:
                        original_filename = "uploaded_video.mp4"
                    temp_dir = tempfile.mkdtemp()
                    tmp_path = os.path.join(temp_dir, original_filename)
                    with open(tmp_path, "wb") as f:
                        f.write(file_obj)
                    run_pipeline(tmp_path)
                    # Optionally, clean up temp file after processing
                    try:
                        os.remove(tmp_path)
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
                    return f"✅ Video analysis complete for uploaded file."
                except Exception as e:
                    return f"❌ Error during video analysis: {str(e)}"
            
            analyze_btn.click(
                fn=analyze_video_local,
                inputs=[video_file],
                outputs=[analysis_output]
            )
            
        except Exception as e:
            gr.Markdown(f"❌ Could not load video analyzer interface: {str(e)}")
            gr.Markdown("Please ensure the Video Analysis MCP Server is running on port 7860.")

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting Video Search Agent...")
    print("📍 Using CLIP model for embeddings:", CLIP_MODEL_NAME)
    print("📍 Ensure the Video Analysis MCP Server is running on port 7860")
    print("📍 Database path should be: output/lancedb")
    # print(f"📍 CLIP model loaded: {'✅' if clip_model_loaded else '❌'}")
    
    demo.launch(
        server_name="localhost",
        server_port=7861,
        share=False,
        debug=True
    )