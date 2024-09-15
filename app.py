import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
import json
import datetime
import networkx as nx
from typing import List, Dict
import io
import base64
import xgboost as xgb

# Advanced Synthetic Data Generation
def generate_synthetic_data(num_transactions: int, fraud_rate: float, transaction_types: List[str]) -> pd.DataFrame:
    data = []
    current_time = datetime.datetime.now()
    
    for _ in range(num_transactions):
        amount = np.random.lognormal(mean=4, sigma=1)
        transaction_type = np.random.choice(transaction_types)
        merchant = np.random.choice(["Amazon", "Walmart", "Target", "Best Buy", "Apple Store", "eBay", "Etsy", "Shopify"])
        location = np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"])
        device = np.random.choice(["Mobile", "Desktop", "Tablet", "Smart TV", "Wearable"])
        is_fraud = np.random.random() < fraud_rate
        
        if is_fraud:
            amount *= np.random.uniform(1.5, 3)
            if np.random.random() < 0.3:
                location = np.random.choice(["Unknown", "International"])
            if np.random.random() < 0.4:
                device = "Unknown"
        
        timestamp = current_time - datetime.timedelta(minutes=np.random.randint(0, 60*24*30))  # Last 30 days
        
        data.append({
            'timestamp': timestamp,
            'amount': amount,
            'transaction_type': transaction_type,
            'merchant': merchant,
            'location': location,
            'device': device,
            'is_fraud': is_fraud
        })
    
    df = pd.DataFrame(data)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    return df

# Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['amount_log'] = np.log1p(df['amount'])
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['transaction_frequency'] = df.groupby('device')['timestamp'].transform('count')
    df['amount_mean'] = df.groupby('device')['amount'].transform('mean')
    df['amount_std'] = df.groupby('device')['amount'].transform('std')
    df['amount_zscore'] = (df['amount'] - df['amount_mean']) / df['amount_std']
    
    cat_columns = ['transaction_type', 'merchant', 'location', 'device']
    df_encoded = pd.get_dummies(df, columns=cat_columns)
    
    return df_encoded

# Advanced Fraud Detection Model
def train_fraud_detection_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    models = {
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'neural_network': nn_model,
        'svm': svm_model,
        'xgboost': xgb_model
    }
    
    for name, model in models.items():
        model.fit(X_scaled, y)
    
    models['scaler'] = scaler
    return models

# Model Evaluation
def evaluate_model(models: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
    X_scaled = models['scaler'].transform(X)
    results = {}
    
    for model_name, model in models.items():
        if model_name != 'scaler':
            y_pred = model.predict(X_scaled)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
            results[model_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_f1_mean': np.mean(cv_scores),
                'cv_f1_std': np.std(cv_scores)
            }
    
    return results

# Advanced Visualizations
def plot_tsne(X: pd.DataFrame, y: pd.Series) -> go.Figure:
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    df_tsne['is_fraud'] = y
    
    fig = px.scatter(df_tsne, x='TSNE1', y='TSNE2', color='is_fraud',
                     title='t-SNE Visualization of Transactions',
                     labels={'is_fraud': 'Is Fraud'},
                     color_discrete_map={0: 'blue', 1: 'red'})
    
    return fig

def plot_feature_importance(model: RandomForestClassifier, feature_names: List[str]) -> go.Figure:
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False).head(20)
    
    fig = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                 title='Top 20 Feature Importances',
                 labels={'importance': 'Importance', 'feature': 'Feature'})
    
    return fig

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Normal', 'Fraud'],
                    y=['Normal', 'Fraud'],
                    title='Confusion Matrix',
                    color_continuous_scale='Blues')
    
    fig.update_layout(width=500, height=500)
    
    return fig

def plot_transaction_network(df: pd.DataFrame) -> go.Figure:
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['location'], row['merchant'], weight=row['amount'])
    
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'{node}<br># of connections: {len(adjacencies)}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Transaction Network',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# Google Gemini API Integration
def analyze_text_with_gemini(text: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": f"Analyze this transaction description for potential fraud indicators and provide a detailed explanation: {text}"}]}]
    }
    params = {'key': ''}  # Replace with your actual API key
    
    response = requests.post(url, headers=headers, params=params, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Function to create a download link for the dataframe
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_data.csv">Download CSV File</a>'
    return href

# Anomaly Detection using Isolation Forest
def detect_anomalies(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return anomalies

# Streamlit Dashboard
def main():
    st.set_page_config(page_title="Advanced Fraud Detection Platform", layout="wide")
    st.title("Financial Fraud Detection Platform")
    
    # Sidebar for data generation parameters
    st.sidebar.header("Data Generation Parameters")
    num_transactions = st.sidebar.slider("Number of Transactions", 1000, 1000000, 100000)
    fraud_rate = st.sidebar.slider("Fraud Rate", 0.001, 0.1, 0.02)
    transaction_types = st.sidebar.multiselect("Transaction Types", 
                                               ["Online", "In-store", "ATM", "Wire Transfer", "Mobile Payment", "Cryptocurrency"],
                                               default=["Online", "In-store", "Mobile Payment"])
    
    # Data generation or upload
    data_option = st.radio("Choose data source:", ("Generate Synthetic Data", "Upload Data"))
    
    if data_option == "Generate Synthetic Data":
        if st.button("Generate Data"):
            data = generate_synthetic_data(num_transactions, fraud_rate, transaction_types)
            st.session_state.data = data
            st.success("Synthetic data generated successfully!")
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success("Data uploaded successfully!")
    
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # Display sample data
        st.subheader("Sample Data")
        st.write(data.head())
        
        # Download link for the data
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        
        # Data visualization
        st.subheader("Data Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(data, x='amount', color='is_fraud', marginal="box",
                               title='Distribution of Transaction Amounts',
                               labels={'amount': 'Amount', 'is_fraud': 'Is Fraud'},
                               color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig)
        
        with col2:
            fig = px.scatter(data, x='amount', y='hour', color='is_fraud', 
                             title='Transaction Amount vs. Hour of Day',
                             labels={'amount': 'Amount', 'hour': 'Hour of Day', 'is_fraud': 'Is Fraud'},
                             color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig)
        
        # Feature Engineering
        df_engineered = engineer_features(data)
        
        # Prepare data for model training
        X = df_engineered.drop(['is_fraud', 'timestamp'], axis=1)
        y = df_engineered['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        models = train_fraud_detection_model(X_train, y_train)
        
        # Evaluate model
        results = evaluate_model(models, X_test, y_test)
        
        st.subheader("Model Performance")
        for model_name, metrics in results.items():
            st.write(f"{model_name.capitalize()}:")
            st.write(f"Precision: {metrics['precision']:.2f}")
            st.write(f"Recall: {metrics['recall']:.2f}")
            st.write(f"F1 Score: {metrics['f1']:.2f}")
            st.write(f"Cross-validation F1 Score: {metrics['cv_f1_mean']:.2f} (± {metrics['cv_f1_std']:.2f})")
            st.write("---")
        
        # Advanced Visualizations
        st.subheader("Advanced Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tsne_plot = plot_tsne(X, y)
            st.plotly_chart(tsne_plot)
        
        with col2:
            feature_importance_plot = plot_feature_importance(models['random_forest'], X.columns)
            st.plotly_chart(feature_importance_plot)
        
        col1, col2 = st.columns(2)
        
        with col1:
            y_pred = models['random_forest'].predict(models['scaler'].transform(X_test))
            cm_plot = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(cm_plot)
        
        with col2:
            network_plot = plot_transaction_network(data)
            st.plotly_chart(network_plot)
        
        # Anomaly Detection
        st.subheader("Anomaly Detection")
        anomalies = detect_anomalies(X)
        anomaly_df = pd.DataFrame({'anomaly': anomalies})
        fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color=anomaly_df['anomaly'],
                         title='Anomaly Detection Results',
                         labels={'color': 'Is Anomaly'},
                         color_discrete_map={1: 'blue', -1: 'red'})
        st.plotly_chart(fig)
        
        # Google Gemini API Integration
        st.subheader("Transaction Description Analysis")
        transaction_description = st.text_input("Enter a transaction description for analysis:")
        if transaction_description:
            analysis = analyze_text_with_gemini(transaction_description)
            st.write("Analysis:", analysis)
        
        # Real-time Fraud Detection Simulation
        st.subheader("Real-time Fraud Detection Simulation")
        if st.button("Generate New Transaction"):
            new_transaction = generate_synthetic_data(1, fraud_rate, transaction_types).iloc[0]
            st.write("New Transaction:")
            st.write(new_transaction)
            
            new_transaction_encoded = engineer_features(pd.DataFrame([new_transaction])).drop(['is_fraud', 'timestamp'], axis=1)
            fraud_probability = models['random_forest'].predict_proba(models['scaler'].transform(new_transaction_encoded))[0][1]
            
            st.write(f"Fraud Probability: {fraud_probability:.2%}")
            if fraud_probability > 0.5:
                st.warning("⚠️ This transaction is flagged as potentially fraudulent!")
            else:
                st.success("✅ This transaction appears to be normal.")

            # Visualize the new transaction in context
            fig = go.Figure()

            # Plot existing transactions
            fig.add_trace(go.Scatter(
                x=data[data['is_fraud'] == 0]['amount'],
                y=data[data['is_fraud'] == 0]['hour'],
                mode='markers',
                name='Normal Transactions',
                marker=dict(color='blue', size=5, opacity=0.5)
            ))
            fig.add_trace(go.Scatter(
                x=data[data['is_fraud'] == 1]['amount'],
                y=data[data['is_fraud'] == 1]['hour'],
                mode='markers',
                name='Fraudulent Transactions',
                marker=dict(color='red', size=5, opacity=0.5)
            ))

            # Plot new transaction
            fig.add_trace(go.Scatter(
                x=[new_transaction['amount']],
                y=[new_transaction['hour']],
                mode='markers',
                name='New Transaction',
                marker=dict(color='green', size=15, symbol='star')
            ))

            fig.update_layout(
                title='New Transaction in Context',
                xaxis_title='Amount',
                yaxis_title='Hour of Day',
                showlegend=True
            )

            st.plotly_chart(fig)

        # Feature Importance for Fraud Detection
        st.subheader("Feature Importance for Fraud Detection")
        feature_importance = models['random_forest'].feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)

        fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h',
                     title='Top 10 Features for Fraud Detection',
                     labels={'importance': 'Importance', 'feature': 'Feature'})
        st.plotly_chart(fig)

        # Fraud Detection Over Time
        st.subheader("Fraud Detection Over Time")
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        fraud_over_time = data.groupby('date')['is_fraud'].mean().reset_index()
        
        fig = px.line(fraud_over_time, x='date', y='is_fraud', 
                      title='Fraud Rate Over Time',
                      labels={'date': 'Date', 'is_fraud': 'Fraud Rate'})
        st.plotly_chart(fig)

        # Geographical Distribution of Transactions
        st.subheader("Geographical Distribution of Transactions")
        location_counts = data['location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']

        fig = px.pie(location_counts, values='count', names='location', 
                     title='Distribution of Transactions by Location')
        st.plotly_chart(fig)

        # Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        corr_matrix = df_engineered[numeric_cols].corr()

        fig = px.imshow(corr_matrix, 
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        title="Feature Correlation Heatmap")
        fig.update_layout(width=800, height=800)
        st.plotly_chart(fig)

        # Model Comparison
        st.subheader("Model Comparison")
        model_names = list(results.keys())
        f1_scores = [metrics['f1'] for metrics in results.values()]
        cv_f1_scores = [metrics['cv_f1_mean'] for metrics in results.values()]

        fig = go.Figure(data=[
            go.Bar(name='F1 Score', x=model_names, y=f1_scores),
            go.Bar(name='CV F1 Score', x=model_names, y=cv_f1_scores)
        ])
        fig.update_layout(barmode='group', title='Model Performance Comparison')
        st.plotly_chart(fig)

        # ROC Curve Comparison
        st.subheader("ROC Curve Comparison")
        fig = go.Figure()
        for model_name, model in models.items():
            if model_name != 'scaler':
                y_pred_proba = model.predict_proba(models['scaler'].transform(X_test))[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{model_name} (AUC = {auc_score:.2f})'))

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
