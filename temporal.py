import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def graph_to_dataframes(G):
    
    nodes_data = []
    for node in G.nodes(data=True):
        node_id = node[0]
        node_attrs = node[1]
        nodes_data.append({
            'node_id': node_id,
            'is_phishing': node_attrs.get('isp', 0) == 1,
            **node_attrs  # Include all other node attributes
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Create edges DataFrame
    edges_data = []
    for u, v, key, data in G.edges(keys=True, data=True):
        # Convert timestamp to hour and day
        timestamp = data.get('timestamp', 0)
        try:
            dt = datetime.fromtimestamp(float(timestamp))
            hour_of_day = dt.hour
            day_of_week = dt.weekday()
        except:
            hour_of_day = 0
            day_of_week = 0
            
        edges_data.append({
            'source': u,
            'target': v,
            'amount': data.get('amount', 0),
            'timestamp': timestamp,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week
        })
    
    edges_df = pd.DataFrame(edges_data)
    
    print(f"Created DataFrames: {len(nodes_df)} nodes, {len(edges_df)} edges")
    return nodes_df, edges_df

def calculate_user_timezones(edges_df, min_transactions=48):
    
    user_timezones = {}
    
    for user_id, user_transactions in edges_df.groupby('source'):
        if len(user_transactions) >= min_transactions:
            hours = user_transactions['hour_of_day'].tolist()
            best_window, min_count = find_least_frequent_6_consecutive(hours)
            timezone = convert_hour_to_timezone(best_window[0])
            user_timezones[user_id] = timezone
    
    print(f"Calculated timezones for {len(user_timezones)} users")
    return user_timezones

def plot_user_timezones(user_timezones):
    # Count users per timezone
    timezone_data = []
    for _, timezone in user_timezones.items():
        timezone_data.append({'timezone': timezone, 'count': 1})
    
    df = pd.DataFrame(timezone_data)
    timezone_counts = df.groupby('timezone').count().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(data=timezone_counts, x='timezone', y='count', 
                     palette='viridis', edgecolor='black', alpha=0.8)
    
    # Customize
    ax.set_xlabel('Timezone (UTC Offset)', fontsize=32)
    ax.set_ylabel('Number of Accounts', fontsize=32)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24)
    
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def find_least_frequent_6_consecutive(hours):
    from collections import Counter
    
    hour_counts = Counter(hours)
    
    min_count = float('inf')
    best_window = None
    
    for start_hour in range(24):
        consecutive_hours = [(start_hour + i) % 24 for i in range(7)]
        total_count = sum(hour_counts.get(h, 0) for h in consecutive_hours)
        
        if total_count < min_count:
            min_count = total_count
            best_window = consecutive_hours
    
    return best_window, min_count

def convert_hour_to_timezone(hour):
    if hour <= 12:
        return hour
    else:
        return hour - 24

def analyze_phishing_transactions_by_timezone(nodes_df, edges_df, user_timezones):
    
    # Get phishing nodes
    phishing_nodes = set(nodes_df[nodes_df['is_phishing']]['node_id'])
    
    phishing_transactions = edges_df[
        (edges_df['target'].isin(phishing_nodes)) &
        (edges_df['source'].isin(user_timezones.keys()))
    ].copy()
    
    # Add timezone information
    phishing_transactions['sender_timezone'] = phishing_transactions['source'].map(user_timezones)
    
    # Calculate adjusted local time
    phishing_transactions['local_hour'] = (
        phishing_transactions['hour_of_day'] + phishing_transactions['sender_timezone']
    ) % 24
    
    
    return phishing_transactions

def analyze_all_transactions_by_timezone(nodes_df, edges_df, user_timezones):

    all_transactions = edges_df[edges_df['source'].isin(user_timezones.keys())]

    all_transactions['sender_timezone'] = all_transactions['source'].map(user_timezones)

    all_transactions['local_hour'] = (
        all_transactions['hour_of_day'] + all_transactions['sender_timezone']
    ) % 24

    return all_transactions

def plot_timezone_analysis(phishing_transactions):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    utc_counts, utc_bins = np.histogram(phishing_transactions['hour_of_day'], bins=24, range=(0, 24))
    utc_percentages = (utc_counts / len(phishing_transactions)) * 100
    utc_counts_0_6 = sum(utc_counts[0:7])
    print("There were ", utc_counts_0_6, " transactions between 0 and 7 UTC")
    utc_percentages_0_6 = (utc_counts_0_6 / len(phishing_transactions)) * 100
    

    local_counts, local_bins = np.histogram(phishing_transactions['local_hour'], bins=24, range=(0, 24))
    local_percentages = (local_counts / len(phishing_transactions)) * 100
    local_counts_0_6 = sum(local_counts[0:7])
    print("There were ", local_counts_0_6, " transactions between 0 and 7 local time")
    local_percentages_0_6 = (local_counts_0_6 / len(phishing_transactions)) * 100
    print("There were ", utc_percentages_0_6, "% of transactions between 0 and 7 UTC" "and", local_percentages_0_6, "% of transactions between 0 and 7 local time")

    ax1.bar(range(24), utc_percentages, alpha=0.7, edgecolor='black', color='blue')
    ax1.set_xlabel('Hour of Day (UTC)')
    ax1.set_ylabel('Percentage of Transactions (%)')
    ax1.set_title('Original Transaction Hours (UTC)')
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(utc_percentages), max(local_percentages)) * 1.1)
    
    ax2.bar(range(24), local_percentages, alpha=0.7, edgecolor='black', color='red')
    ax2.set_xlabel('Hour of Day (Local Time)')
    ax2.set_ylabel('Percentage of Transactions (%)')
    ax2.set_title('Adjusted Transaction Hours (Local Time)')
    ax2.set_xticks(range(24))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(max(utc_percentages), max(local_percentages)) * 1.1)
    
    plt.tight_layout()
    plt.show()
    
    
    return phishing_transactions


if __name__ == "__main__":
    G = load_pickle('./phishing_subgraph.pkl')
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
   
    nodes_df, edges_df = graph_to_dataframes(G)
    
    user_timezones = calculate_user_timezones(edges_df)
    plot_user_timezones(user_timezones)

    all_transactions = analyze_all_transactions_by_timezone(nodes_df, edges_df, user_timezones)

    plot_timezone_analysis(all_transactions)
    
    phishing_transactions = analyze_phishing_transactions_by_timezone(nodes_df, edges_df, user_timezones)
    
    plot_timezone_analysis(phishing_transactions)
    
    
    # Save results
    phishing_transactions.to_csv('phishing_transactions_with_timezones.csv', index=False)
    print("\nSaved detailed results to 'phishing_transactions_with_timezones.csv'")
    