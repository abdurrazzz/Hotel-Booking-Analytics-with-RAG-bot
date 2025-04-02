# app.py - Main Streamlit application
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from datetime import datetime
import json

# Set page configuration
st.set_page_config(page_title="Hotel Booking Analytics", layout="wide")

# Constants
API_URL = "http://localhost:8000"

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\mabdu\OneDrive\Documents\RAG-bot\hotel_bookings.csv')
        # Convert date to datetime
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Analytics", "Q&A System"])

# Load the data
df = load_data()

if df is not None:
    if page == "Dashboard":
        st.title("Hotel Booking Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Bookings", f"{len(df):,}")
        with col2:
            cancellation_rate = df['is_canceled'].mean() * 100
            st.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")
        with col3:
            avg_price = df['adr'].mean()
            st.metric("Avg. Daily Rate", f"${avg_price:.2f}")
        with col4:
            avg_lead_time = df['lead_time'].mean()
            st.metric("Avg. Lead Time", f"{avg_lead_time:.1f} days")
        
        # Visualization section
        st.subheader("Booking Trends")
        
        # Monthly booking trend
        df['month_year'] = pd.to_datetime(df['reservation_status_date']).dt.strftime('%Y-%m')
        monthly_bookings = df.groupby('month_year').size().reset_index(name='count')
        
        # Plot
        fig = px.line(monthly_bookings, x='month_year', y='count', 
                      title='Monthly Booking Trends',
                      labels={'count': 'Number of Bookings', 'month_year': 'Month-Year'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by hotel type
        col1, col2 = st.columns(2)
        with col1:
            hotel_counts = df['hotel'].value_counts().reset_index()
            hotel_counts.columns = ['Hotel Type', 'Count']
            fig = px.pie(hotel_counts, values='Count', names='Hotel Type', 
                         title='Distribution by Hotel Type')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            status_counts = df['reservation_status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig = px.pie(status_counts, values='Count', names='Status', 
                         title='Reservation Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Analytics":
        st.title("Advanced Analytics")
        
        # Create tabs for different analytics views
        tab1, tab2, tab3, tab4 = st.tabs(["Revenue Analysis", "Cancellation Analysis", 
                                          "Guest Demographics", "Booking Patterns"])
        
        with tab1:
            st.subheader("Revenue Analysis")
            
            # Monthly revenue trends
            df['revenue'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])
            df['month_year'] = pd.to_datetime(df['reservation_status_date']).dt.strftime('%Y-%m')
            
            # Filter only non-canceled bookings for revenue analysis
            non_canceled = df[df['is_canceled'] == 0]
            monthly_revenue = non_canceled.groupby('month_year')['revenue'].sum().reset_index()
            
            fig = px.line(monthly_revenue, x='month_year', y='revenue', 
                          title='Monthly Revenue Trends',
                          labels={'revenue': 'Revenue ($)', 'month_year': 'Month-Year'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Revenue by market segment
            segment_revenue = non_canceled.groupby('market_segment')['revenue'].sum().reset_index()
            segment_revenue = segment_revenue.sort_values('revenue', ascending=False)
            
            fig = px.bar(segment_revenue, x='market_segment', y='revenue',
                        title='Revenue by Market Segment',
                        labels={'revenue': 'Revenue ($)', 'market_segment': 'Market Segment'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Cancellation Analysis")
            
            # Cancellation rate by month
            df['month'] = pd.to_datetime(df['reservation_status_date']).dt.month_name()
            cancellation_by_month = df.groupby('month')['is_canceled'].mean().reset_index()
            cancellation_by_month['cancellation_rate'] = cancellation_by_month['is_canceled'] * 100
            
            # Sort months chronologically
            months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            cancellation_by_month['month'] = pd.Categorical(
                cancellation_by_month['month'], categories=months_order, ordered=True)
            cancellation_by_month = cancellation_by_month.sort_values('month')
            
            fig = px.bar(cancellation_by_month, x='month', y='cancellation_rate',
                        title='Cancellation Rate by Month',
                        labels={'cancellation_rate': 'Cancellation Rate (%)', 'month': 'Month'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Lead time vs cancellation correlation
            lead_time_bins = [0, 30, 60, 90, 180, 365, float('inf')]
            lead_time_labels = ['0-30', '31-60', '61-90', '91-180', '181-365', '365+']
            
            df['lead_time_group'] = pd.cut(df['lead_time'], bins=lead_time_bins, labels=lead_time_labels)
            lead_time_cancel = df.groupby('lead_time_group')['is_canceled'].mean().reset_index()
            lead_time_cancel['cancellation_rate'] = lead_time_cancel['is_canceled'] * 100
            
            fig = px.bar(lead_time_cancel, x='lead_time_group', y='cancellation_rate',
                        title='Cancellation Rate by Lead Time',
                        labels={'cancellation_rate': 'Cancellation Rate (%)', 
                               'lead_time_group': 'Lead Time (days)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Guest Demographics")
            
            # Top 10 countries
            country_counts = df['country'].value_counts().nlargest(10).reset_index()
            country_counts.columns = ['Country', 'Count']
            
            fig = px.bar(country_counts, x='Country', y='Count',
                        title='Top 10 Guest Countries',
                        labels={'Count': 'Number of Bookings', 'Country': 'Country Code'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Customer type distribution
            customer_counts = df['customer_type'].value_counts().reset_index()
            customer_counts.columns = ['Customer Type', 'Count']
            
            fig = px.pie(customer_counts, values='Count', names='Customer Type',
                        title='Customer Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Booking Patterns")
            
            # Booking lead time distribution
            fig = px.histogram(df, x='lead_time', nbins=50,
                              title='Booking Lead Time Distribution',
                              labels={'lead_time': 'Lead Time (days)', 'count': 'Number of Bookings'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution channel analysis
            channel_counts = df.groupby(['distribution_channel', 'market_segment']).size().reset_index(name='count')
            
            fig = px.bar(channel_counts, x='distribution_channel', y='count', color='market_segment',
                        title='Bookings by Distribution Channel and Market Segment',
                        labels={'count': 'Number of Bookings', 'distribution_channel': 'Distribution Channel'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Average daily rate by room type
            room_adr = df.groupby('reserved_room_type')['adr'].mean().reset_index()
            room_adr = room_adr.sort_values('adr', ascending=False)
            
            fig = px.bar(room_adr, x='reserved_room_type', y='adr',
                        title='Average Daily Rate by Room Type',
                        labels={'adr': 'Average Daily Rate ($)', 'reserved_room_type': 'Room Type'})
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Q&A System":
        st.title("Q&A System")
        
        st.write("""
        Ask questions about hotel booking data! Examples:
        - What is the average daily rate for Resort Hotels?
        - What month has the highest cancellation rate?
        - How does lead time affect booking cancellations?
        - What is the most common distribution channel?
        """)
        
        # User input
        question = st.text_input("Ask a question about hotel bookings:")
        
        if st.button("Get Answer"):
            if question:
                try:
                    # Call the API
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Answer:")
                        st.write(result["answer"])
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")
            else:
                st.warning("Please enter a question.")
else:
    st.error("Failed to load data. Please ensure the hotel_bookings.csv file is available.")