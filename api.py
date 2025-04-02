# api.py - FastAPI Backend for Hotel Booking Analytics
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from typing import Dict, Any, List
import os

# Initialize FastAPI app
app = FastAPI(title="Hotel Booking Analytics API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class AnalyticsRequest(BaseModel):
    report_type: str

class QuestionRequest(BaseModel):
    question: str

# Load and prepare data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\mabdu\OneDrive\Documents\RAG-bot\hotel_bookings.csv')
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
        
        # Calculate revenue
        df['revenue'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Initialize ChromaDB
def init_chroma():
    # Use the sentence transformer embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create the client and collection
    client = chromadb.Client()
    
    # Check if collection already exists and recreate it
    try:
        client.delete_collection("hotel_bookings")
    except:
        pass
    
    collection = client.create_collection(
        name="hotel_bookings",
        embedding_function=embedding_function
    )
    
    return client, collection

# Initialize LLM
def init_llm():
    llm = LlamaCpp(
        model_path="llama-3-8b-instruct.gguf",  # Path to your downloaded model
        temperature=0.1,
        max_tokens=1024,
        top_p=0.9,
        n_ctx=2048,
        verbose=False
    )
    
    prompt_template = """
    You are a helpful assistant that answers questions about hotel booking data. 
    
    Here is some context information about the hotel bookings data:
    {context}
    
    Based on this context, please answer the following question. If you cannot answer the question
    based on the provided context, please say "I don't have enough information to answer that question."
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain

# Create document chunks for RAG
def create_chunks(df):
    chunks = []
    
    # Overall statistics summary
    total_bookings = len(df)
    avg_adr = df['adr'].mean()
    cancellation_rate = df['is_canceled'].mean() * 100
    avg_lead_time = df['lead_time'].mean()
    
    summary = f"""
    The hotel booking dataset contains {total_bookings} bookings.
    The average daily rate is ${avg_adr:.2f}.
    The overall cancellation rate is {cancellation_rate:.2f}%.
    The average lead time for bookings is {avg_lead_time:.1f} days.
    """
    chunks.append(("overall_summary", summary))
    
    # Hotel type stats
    for hotel_type in df['hotel'].unique():
        hotel_df = df[df['hotel'] == hotel_type]
        hotel_info = f"""
        Hotel type: {hotel_type}
        Number of bookings: {len(hotel_df)}
        Average daily rate: ${hotel_df['adr'].mean():.2f}
        Cancellation rate: {hotel_df['is_canceled'].mean() * 100:.2f}%
        Average lead time: {hotel_df['lead_time'].mean():.1f} days
        """
        chunks.append((f"{hotel_type}_stats", hotel_info))
    
    # Monthly stats
    df['month'] = pd.to_datetime(df['reservation_status_date']).dt.month_name()
    for month in df['month'].unique():
        month_df = df[df['month'] == month]
        month_info = f"""
        Month: {month}
        Number of bookings: {len(month_df)}
        Average daily rate: ${month_df['adr'].mean():.2f}
        Cancellation rate: {month_df['is_canceled'].mean() * 100:.2f}%
        """
        chunks.append((f"{month}_stats", month_info))
    
    # Distribution channel stats
    for channel in df['distribution_channel'].unique():
        channel_df = df[df['distribution_channel'] == channel]
        channel_info = f"""
        Distribution channel: {channel}
        Number of bookings: {len(channel_df)}
        Percentage of total bookings: {(len(channel_df) / len(df)) * 100:.2f}%
        Average daily rate: ${channel_df['adr'].mean():.2f}
        Cancellation rate: {channel_df['is_canceled'].mean() * 100:.2f}%
        """
        chunks.append((f"{channel}_stats", channel_info))
    
    # Lead time analysis
    lead_time_bins = [0, 30, 60, 90, 180, 365, float('inf')]
    lead_time_labels = ['0-30 days', '31-60 days', '61-90 days', '91-180 days', '181-365 days', 'over 365 days']
    df['lead_time_group'] = pd.cut(df['lead_time'], bins=lead_time_bins, labels=lead_time_labels)
    
    for lead_group in df['lead_time_group'].unique():
        lead_df = df[df['lead_time_group'] == lead_group]
        lead_info = f"""
        Lead time group: {lead_group}
        Number of bookings: {len(lead_df)}
        Cancellation rate: {lead_df['is_canceled'].mean() * 100:.2f}%
        Average daily rate: ${lead_df['adr'].mean():.2f}
        """
        chunks.append((f"{lead_group}_stats", lead_info))
    
    # Revenue analysis for non-canceled bookings
    non_canceled = df[df['is_canceled'] == 0]
    revenue_info = f"""
    Total revenue from completed bookings: ${non_canceled['revenue'].sum():.2f}
    Average revenue per booking: ${non_canceled['revenue'].mean():.2f}
    """
    chunks.append(("revenue_summary", revenue_info))
    
    # Country analysis
    top_countries = df['country'].value_counts().nlargest(5)
    country_info = "Top 5 countries by number of bookings:\n"
    for country, count in top_countries.items():
        country_info += f"{country}: {count} bookings ({(count/len(df))*100:.2f}%)\n"
    chunks.append(("country_summary", country_info))
    
    return chunks

# Load data and initialize components on startup
df = None
collection = None
llm_chain = None

@app.on_event("startup")
async def startup_event():
    global df, collection, llm_chain
    
    # Load data
    df = load_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to load hotel booking data")
    
    # Initialize ChromaDB
    client, collection = init_chroma()
    
    # Create and add chunks to collection
    chunks = create_chunks(df)
    
    # Add chunks to the collection
    ids = [chunk[0] for chunk in chunks]
    documents = [chunk[1] for chunk in chunks]
    collection.add(
        ids=ids,
        documents=documents
    )
    
    # Initialize LLM
    try:
        llm_chain = init_llm()
    except Exception as e:
        print(f"Warning: LLM initialization failed: {e}")
        print("Q&A functionality will be limited to simple responses")

# Analytics endpoint
@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    global df
    
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if request.report_type == "revenue":
        # Revenue analysis
        revenue_data = df[df['is_canceled'] == 0].copy()
        monthly_revenue = revenue_data.groupby(pd.Grouper(key='reservation_status_date', freq='M'))['revenue'].sum()
        
        return {
            "report_type": "revenue",
            "total_revenue": float(revenue_data['revenue'].sum()),
            "avg_revenue_per_booking": float(revenue_data['revenue'].mean()),
            "monthly_data": {str(date.date()): float(value) for date, value in monthly_revenue.items()}
        }
    
    elif request.report_type == "cancellations":
        # Cancellation analysis
        monthly_cancel_rate = df.groupby(pd.Grouper(key='reservation_status_date', freq='M'))['is_canceled'].mean()
        
        return {
            "report_type": "cancellations",
            "overall_rate": float(df['is_canceled'].mean()),
            "monthly_rates": {str(date.date()): float(value) for date, value in monthly_cancel_rate.items()}
        }
    
    elif request.report_type == "geography":
        # Geographic analysis
        country_counts = df['country'].value_counts()
        
        return {
            "report_type": "geography",
            "country_distribution": {country: int(count) for country, count in country_counts.items()}
        }
    
    elif request.report_type == "lead_time":
        # Lead time analysis
        return {
            "report_type": "lead_time",
            "avg_lead_time": float(df['lead_time'].mean()),
            "median_lead_time": float(df['lead_time'].median()),
            "lead_time_distribution": {
                "0-30 days": int(df[df['lead_time'] <= 30]['lead_time'].count()),
                "31-60 days": int(df[(df['lead_time'] > 30) & (df['lead_time'] <= 60)]['lead_time'].count()),
                "61-90 days": int(df[(df['lead_time'] > 60) & (df['lead_time'] <= 90)]['lead_time'].count()),
                "91-180 days": int(df[(df['lead_time'] > 90) & (df['lead_time'] <= 180)]['lead_time'].count()),
                "181-365 days": int(df[(df['lead_time'] > 180) & (df['lead_time'] <= 365)]['lead_time'].count()),
                "over 365 days": int(df[df['lead_time'] > 365]['lead_time'].count()),
            }
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown report type: {request.report_type}")

# Q&A endpoint
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global collection, llm_chain, df
    
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized")
    
    try:
        # Query the collection for relevant chunks
        results = collection.query(
            query_texts=[request.question],
            n_results=3
        )
        
        context = "\n".join(results['documents'][0])
        
        # If LLM is available, use it for answering
        if llm_chain is not None:
            response = llm_chain.run({
                "context": context,
                "question": request.question
            })
            return {"answer": response}
        
        # Fallback responses for common questions if LLM is not available
        if "average daily rate" in request.question.lower():
            if "resort" in request.question.lower():
                avg_rate = df[df['hotel'] == 'Resort Hotel']['adr'].mean()
                return {"answer": f"The average daily rate for Resort Hotels is ${avg_rate:.2f}."}
            else:
                avg_rate = df['adr'].mean()
                return {"answer": f"The overall average daily rate is ${avg_rate:.2f}."}
        
        elif "cancellation rate" in request.question.lower():
            cancel_rate = df['is_canceled'].mean() * 100
            return {"answer": f"The overall cancellation rate is {cancel_rate:.2f}%."}
        
        elif "distribution channel" in request.question.lower():
            top_channel = df['distribution_channel'].value_counts().idxmax()
            return {"answer": f"The most common distribution channel is {top_channel}."}
        
        else:
            return {"answer": "I found this information that might help: " + context}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hotel Booking Analytics API is running"}

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)