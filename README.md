# Hotel Booking Analytics System

A complete system for analyzing hotel booking data with a Streamlit frontend, FastAPI backend, and RAG-based Q&A capabilities.

## System Architecture
The system consists of three main components:

1. **Streamlit Frontend**:
   - Provides an interactive dashboard for data visualization and insights.
   - Enables users to query the RAG-based Q&A system.

2. **FastAPI Backend**:
   - Serves as a REST API for analytics and question answering.
   - Facilitates communication between the frontend and the RAG Q&A system.

3. **RAG-based Q&A System**:
   - Utilizes **LangChain** to orchestrate retrieval-augmented generation (RAG).
   - Stores embeddings in **ChromaDB** for efficient document retrieval.
   - Leverages **Llama 3** for intelligent and context-aware question answering.


## Features

### Analytics & Reporting
- Revenue trends and analysis
- Cancellation rate patterns
- Guest demographics and geographic analysis
- Booking lead time statistics
- Room type and pricing analysis

### Question Answering
- ChromaDB for storing data embeddings
- Llama 3 for natural language understanding
- Answer questions about booking trends, metrics, and patterns

### API Endpoints
- `POST /analytics`: Returns detailed analytics reports
- `POST /ask`: Answers booking-related questions using RAG

## Getting Started

### Prerequisites
- Python 3.8+
- Llama 3 model (8B instruct version)
- Hotel bookings dataset (CSV)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hotel-booking-analytics.git
cd hotel-booking-analytics
```

2. Set up the environment:
```bash
bash setup.sh
```

3. Download the Llama 3 model:
   - Get the model from [HuggingFace](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF)
   - Save it as `llama-3-8b-instruct.gguf` in the project directory

4. Place your `hotel_bookings.csv` file in the project directory

### Running the Application

Start both the API and Streamlit app:
```bash
bash setup.sh
```

Or start them separately:
```bash
# Start the API
uvicorn api:app --reload

# Start Streamlit
streamlit run app.py
```

Visit:
- Streamlit dashboard: http://localhost:8501
- API documentation: http://localhost:8000/docs

## API Usage

### Analytics Endpoint
```python
import requests

response = requests.post(
    "http://localhost:8000/analytics",
    json={"report_type": "revenue"}
)
print(response.json())
```

### Question Answering Endpoint
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is the average daily rate for Resort Hotels?"}
)
print(response.json())
```

## License

MIT License
