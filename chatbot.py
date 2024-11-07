import torch
from transformers import pipeline
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import gradio as gr
from typing import List, Tuple

class LlamaAirbnbChatbot:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        print("Initializing Llama chatbot...")
        
        # Initialize the LLM pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token="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"
        )
        
        # Initialize sentence transformer for embeddings
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load FAISS index directly
        print("Loading vector store...")
        try:
            self.index = faiss.read_index("unified_faiss_index/index.faiss")
            with open("unified_faiss_index/documents.json", "r") as f:
                self.documents = json.load(f)
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            raise
        
        # Initialize conversation history
        self.reset_conversation()
    
    def reset_conversation(self):
        """Reset the conversation history to initial state"""
        self.conversation_history = [
            {
                "role": "system",
                "content": """You are an Airbnb property assistant. You help users find 
                information about properties, analyze reviews, and understand pricing. 
                Always provide clear, concise, and helpful responses based on the 
                context provided."""
            }
        ]
    
    def get_relevant_docs(self, query, k=3):
        """Retrieve relevant documents using FAISS"""
        query_vector = self.embedding_model.encode([query])[0]
        D, I = self.index.search(np.array([query_vector], dtype=np.float32), k)
        return [self.documents[i] for i in I[0]]
    
    def format_context(self, relevant_docs):
        """Format retrieved documents into a string"""
        return "\n\n".join([doc for doc in relevant_docs])
    
    def get_response(self, query: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        try:
            # Retrieve relevant documents
            relevant_docs = self.get_relevant_docs(query)
            context = self.format_context(relevant_docs)
            
            # Add user query to conversation
            self.conversation_history.append({
                "role": "user",
                "content": f"{context}\nQuestion: {query}"
            })
            
            # Generate response
            outputs = self.pipe(
                self.conversation_history,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Extract and clean response
            response = outputs[0]["generated_text"]
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[:1] + self.conversation_history[-9:]
            
            # Update Gradio chat history
            chat_history.append((query, response))
            
            return response, chat_history
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            chat_history.append((query, error_message))
            return error_message, chat_history

def create_gradio_interface():
    # Initialize chatbot
    chatbot = LlamaAirbnbChatbot()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .chatbot {
        height: 600px;
        overflow-y: auto;
    }
    .message {
        padding: 10px;
        margin: 5px;
        border-radius: 15px;
    }
    .user {
        background-color: #e3f2fd;
    }
    .bot {
        background-color: #f5f5f5;
    }
    """
    
    # Create Gradio interface
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="blue",
        spacing_size="sm",
        radius_size="lg",
        font=["Arial", "sans-serif"]
    )) as demo:
        gr.Markdown(
            """
            # 🏠 Airbnb Property Assistant
            Your AI guide for finding and understanding Airbnb properties.
            """
        )
        
        chatbot_interface = gr.Chatbot(
            [],
            elem_id="chatbot",
            avatar_images=("👤", "🏠"),
            height=500,
            bubble_full_width=False,
        )
        
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Ask me anything about the properties...",
                container=False,
                scale=8
            )
            submit = gr.Button(
                "Send",
                variant="primary",
                scale=1,
                min_width=100
            )
        
        with gr.Row():
            clear = gr.Button("Clear Chat", variant="secondary")
            examples = gr.Examples(
                examples=[
                    "What properties are available in downtown?",
                    "Tell me about properties with good reviews",
                    "Show me properties under $100 per night",
                    "What are the best-rated properties?",
                    "Tell me about properties with good location ratings"
                ],
                inputs=msg,
                label="Try these examples"
            )
        
        # State
        state = gr.State([])
        
        # Event handlers
        def user_input(user_message, history):
            return "", *chatbot.get_response(user_message, history)
        
        msg.submit(user_input, [msg, state], [msg, chatbot_interface, state])
        submit.click(user_input, [msg, state], [msg, chatbot_interface, state])
        
        def clear_chat():
            chatbot.reset_conversation()
            return [], []
        
        clear.click(clear_chat, None, [chatbot_interface, state])
        
        gr.Markdown(
            """
            ### 📝 Notes:
            - Ask about property details, prices, locations, and reviews
            - Get information about amenities and host details
            - Compare different properties
            - Understand pricing trends
            """
        )
    
    return demo

def main():
    demo = create_gradio_interface()
    demo.queue()  # Enable queuing for better handling of multiple users
    demo.launch(
        # share=True,  # Set to False if you don't want to share publicly
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )

if __name__ == "__main__":
    main()