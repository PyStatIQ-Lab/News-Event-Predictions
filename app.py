import streamlit as st
import datetime
from collections import defaultdict, Counter
import json
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
st.set_page_config(layout="wide", page_title="Stock News Analytics Dashboard", page_icon="📈")

# Define event types with keywords
EVENT_TYPES = [
    ("acquisition", ["acquire", "acquisition", "buys", "takeover", "stake buy", "stake sale"]),
    ("partnership", ["partner", "partners", "teams up", "collaborat", "joint venture", "joins hands"]),
    ("agreement", ["agreement", "signs", "deal", "contract", "pact"]),
    ("investment", ["invest", "funding", "raise capital", "infuse"]),
    ("launch", ["launch", "introduce", "release", "unveil"]),
    ("expansion", ["expand", "expansion", "new facility", "new plant"]),
    ("award", ["award", "recognize", "prize"]),
    ("leadership", ["appoint", "hire", "resign", "exit", "join as", "takes over"]),
    ("financial", ["results", "earnings", "profit", "revenue", "dividend"]),
    ("regulatory", ["regulator", "sebi", "rbi", "government", "approval", "clearance"])
]

# Cache news data to avoid repeated API calls
@st.cache_data(ttl=3600)  # Refresh every hour
def fetch_news_data():
    url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=500"
    response = requests.get(url)
    return response.json()['data']

# Cache stock data
@st.cache_data(ttl=3600)
def get_stock_data(symbol, exchange):
    suffix_map = {'nse': '.NS', 'bse': '.BO'}
    suffix = suffix_map.get(exchange.lower(), '')
    ticker = symbol + suffix
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty:
            return None, None, None, None
        
        start_price = hist['Close'].iloc[0]
        current_price = hist['Close'].iloc[-1]
        pct_change = ((current_price - start_price) / start_price) * 100
        
        if pct_change > 5:
            trend = "strong_up"
        elif pct_change > 1:
            trend = "moderate_up"
        elif pct_change < -5:
            trend = "strong_down"
        elif pct_change < -1:
            trend = "moderate_down"
        else:
            trend = "neutral"
            
        return current_price, pct_change, trend, hist
    
    except Exception as e:
        return None, None, None, None

def extract_event_type(headline):
    headline_lower = headline.lower()
    for event_type, keywords in EVENT_TYPES:
        for kw in keywords:
            if kw in headline_lower:
                return event_type
    return "other"

def analyze_stock_news_correlation(events, history):
    if history is None:
        return 0
        
    event_dates = [date for _, date in events]
    event_dates = pd.Series(event_dates, name='event_date')
    prices = history[['Close']].reset_index()
    
    results = []
    for date in event_dates:
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)
        
        prev_price = prices[prices['Date'] == prev_day]['Close'].values
        next_price = prices[prices['Date'] == next_day]['Close'].values
        
        if len(prev_price) > 0 and len(next_price) > 0:
            change = ((next_price[0] - prev_price[0]) / prev_price[0]) * 100
            results.append(change)
    
    if results:
        return sum(results) / len(results)
    return 0

def generate_word_cloud(news_data):
    headlines = [item['headline'] for item in news_data]
    text = " ".join(headlines)
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='viridis',
                          max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def generate_sentiment_analysis(news_data):
    sentiments = []
    for item in news_data:
        headline = item['headline'].lower()
        positive_words = ['growth', 'profit', 'gain', 'surge', 'rise', 'upgrade', 'buy', 'strong', 'win', 'success']
        negative_words = ['fall', 'loss', 'decline', 'drop', 'cut', 'sell', 'weak', 'risk', 'warn', 'fail']
        
        positive_count = sum(headline.count(word) for word in positive_words)
        negative_count = sum(headline.count(word) for word in negative_words)
        
        if positive_count > negative_count:
            sentiments.append('Positive')
        elif negative_count > positive_count:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
    
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, 
                 names=sentiment_counts.index, 
                 title='News Sentiment Distribution',
                 color_discrete_sequence=px.colors.sequential.Viridis)
    return fig

def generate_top_companies_chart(news_data):
    companies = []
    for item in news_data:
        if 'linkedScrips' in item:
            for scrip in item['linkedScrips']:
                companies.append(scrip['symbol'])
    
    if not companies:
        return None
    
    company_counts = pd.Series(companies).value_counts().head(10)
    fig = px.bar(company_counts, x=company_counts.values, y=company_counts.index,
                 orientation='h', title='Top Companies in News',
                 color=company_counts.values,
                 color_continuous_scale='Viridis')
    fig.update_layout(yaxis_title='Company', xaxis_title='News Count')
    return fig

def generate_event_timeline(news_data):
    events = []
    for item in news_data:
        try:
            date = datetime.datetime.strptime(
                item['publishedAt'].replace('Z', ''),
                "%Y-%m-%dT%H:%M:%S.%f"
            ).date()
            events.append({'date': date, 'headline': item['headline']})
        except:
            continue
    
    if not events:
        return None
    
    events_df = pd.DataFrame(events)
    events_df = events_df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(events_df, x='date', y='count', 
                  title='News Events Timeline',
                  markers=True)
    fig.update_traces(line_color='#5e35b1', marker_color='#9c27b0')
    return fig

def main():
    # Load news data
    st.sidebar.title("Dashboard Settings")
    st.sidebar.info("This dashboard analyzes stock news and predicts future events based on historical patterns")
    
    # Load data with progress indicator
    with st.spinner('Fetching latest news data...'):
        news_data = fetch_news_data()
    
    st.title("📈 Stock News Analytics Dashboard")
    st.caption("Analyzing market news to predict corporate events and stock movements")
    
    # Overview metrics
    st.subheader("Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total News Articles", len(news_data))
    unique_companies = len(set(scrip['symbol'] for item in news_data if 'linkedScrips' in item for scrip in item['linkedScrips']))
    col2.metric("Companies Covered", unique_companies)
    
    # Get Nifty and Sensex data
    nifty = yf.Ticker("^NSEI")
    nifty_data = nifty.history(period='1d')
    nifty_change = ((nifty_data['Close'][0] - nifty_data['Open'][0]) / nifty_data['Open'][0]) * 100
    
    sensex = yf.Ticker("^BSESN")
    sensex_data = sensex.history(period='1d')
    sensex_change = ((sensex_data['Close'][0] - sensex_data['Open'][0]) / sensex_data['Open'][0]) * 100
    
    col3.metric("Nifty 50", f"{nifty_data['Close'][0]:.2f}", f"{nifty_change:.2f}%")
    col4.metric("BSE Sensex", f"{sensex_data['Close'][0]:.2f}", f"{sensex_change:.2f}%")
    
    # Top row visualizations
    st.subheader("News Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(generate_word_cloud(news_data))
        st.caption("Word Cloud of Most Frequent Terms in News Headlines")
    
    with col2:
        st.plotly_chart(generate_sentiment_analysis(news_data), use_container_width=True)
    
    # Middle row visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        top_companies = generate_top_companies_chart(news_data)
        if top_companies:
            st.plotly_chart(top_companies, use_container_width=True)
        else:
            st.warning("No company data available for visualization")
    
    with col2:
        timeline = generate_event_timeline(news_data)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
        else:
            st.warning("No date data available for timeline")
    
    # Event prediction section
    st.subheader("Event Prediction & Stock Analysis")
    
    # Process news data for predictions
    company_events = defaultdict(list)
    current_date = datetime.datetime(2025, 6, 19)
    
    for item in news_data:
        try:
            pub_date = datetime.datetime.strptime(
                item['publishedAt'].replace('Z', ''),
                "%Y-%m-%dT%H:%M:%S.%f"
            )
        except:
            continue
        
        if not item.get('linkedScrips'):
            continue
            
        event_type = extract_event_type(item['headline'])
        if event_type == "other":
            continue
            
        for company in item['linkedScrips']:
            symbol = company['symbol']
            exchange = company['exchange']
            company_events[(symbol, exchange)].append((event_type, pub_date))
    
    # Create predictions
    predictions = []
    for (symbol, exchange), events in company_events.items():
        recent_events = [
            event_type for event_type, date in events
            if (current_date - date).days <= 30
        ]
        
        if not recent_events:
            continue
            
        event_counts = Counter(recent_events)
        most_common = event_counts.most_common(1)[0][0]
        confidence = min(100, event_counts[most_common] * 20)
        price, pct_change, trend, history = get_stock_data(symbol, exchange)
        impact = analyze_stock_news_correlation(events, history) if history is not None else 0
        
        predictions.append({
            "symbol": symbol,
            "exchange": exchange,
            "predicted_event": most_common,
            "confidence": confidence,
            "recent_occurrences": event_counts[most_common],
            "current_price": price,
            "price_change_pct": pct_change,
            "price_trend": trend,
            "news_impact_pct": impact
        })
    
    # Display predictions in a table
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Add trend icons
        trend_icons = {
            "strong_up": "🚀",
            "moderate_up": "📈",
            "neutral": "➖",
            "moderate_down": "📉",
            "strong_down": "💥"
        }
        df['trend_icon'] = df['price_trend'].map(trend_icons)
        
        # Format columns
        df['current_price'] = df['current_price'].apply(lambda x: f"₹{x:.2f}" if x else "N/A")
        df['price_change_pct'] = df['price_change_pct'].apply(lambda x: f"{x:.2f}%" if x else "N/A")
        df['news_impact_pct'] = df['news_impact_pct'].apply(lambda x: f"{x:.2f}%" if x else "N/A")
        
        # Color confidence column
        def color_confidence(val):
            color = 'green' if val > 70 else 'orange' if val > 40 else 'red'
            return f'color: {color}; font-weight: bold'
        
        # Display table
        st.dataframe(df[['symbol', 'exchange', 'predicted_event', 'confidence', 
                         'recent_occurrences', 'current_price', 'price_change_pct',
                         'trend_icon', 'news_impact_pct']].rename(columns={
                             'symbol': 'Symbol',
                             'exchange': 'Exchange',
                             'predicted_event': 'Predicted Event',
                             'confidence': 'Confidence',
                             'recent_occurrences': 'Recent Events',
                             'current_price': 'Price',
                             'price_change_pct': '3M Change',
                             'trend_icon': 'Trend',
                             'news_impact_pct': 'News Impact'
                         }).style.applymap(color_confidence, subset=['Confidence']),
                    height=400, use_container_width=True)
    else:
        st.warning("No predictions available based on current news data")
    
    # Detailed news section
    st.subheader("Latest Market News")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_company = st.selectbox("Filter by Company", 
                                        options=['All'] + sorted(list(set(
                                            scrip['symbol'] for item in news_data 
                                            if 'linkedScrips' in item 
                                            for scrip in item['linkedScrips']
                                        ))))
    with col2:
        selected_event = st.selectbox("Filter by Event Type", 
                                     options=['All'] + [et[0] for et in EVENT_TYPES])
    
    # Display news cards
    for i, item in enumerate(news_data[:10]):  # Show top 10 news items
        # Apply filters
        if selected_company != 'All':
            if 'linkedScrips' not in item or not any(scrip['symbol'] == selected_company for scrip in item['linkedScrips']):
                continue
        
        event_type = extract_event_type(item['headline'])
        if selected_event != 'All' and event_type != selected_event:
            continue
        
        # Create card
        with st.expander(f"{item['headline']}", expanded=(i==0)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if 'thumbnailImage' in item and item['thumbnailImage']:
                    st.image(item['thumbnailImage']['url'], width=200)
            
            with col2:
                # Format date
                try:
                    pub_date = datetime.datetime.strptime(
                        item['publishedAt'].replace('Z', ''),
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    st.caption(f"Published: {pub_date.strftime('%b %d, %Y %H:%M')}")
                except:
                    st.caption("Published: N/A")
                
                # Show event type with color coding
                event_color = {
                    'acquisition': 'blue',
                    'partnership': 'green',
                    'agreement': 'orange',
                    'investment': 'purple',
                    'launch': 'red',
                    'expansion': 'teal',
                    'award': 'gold',
                    'leadership': 'pink',
                    'financial': 'brown',
                    'regulatory': 'gray'
                }.get(event_type, 'black')
                
                st.markdown(f"**Event Type:** <span style='color:{event_color}; font-weight:bold'>{event_type.title()}</span>", 
                            unsafe_allow_html=True)
                
                st.write(item['summary'])
                
                # Show linked companies
                if 'linkedScrips' in item and item['linkedScrips']:
                    companies = ", ".join(scrip['symbol'] for scrip in item['linkedScrips'])
                    st.markdown(f"**Related Companies:** {companies}")
                
                st.markdown(f"[Read full article]({item['contentUrl']})")

if __name__ == "__main__":
    main()
