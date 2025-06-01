## https://github.com/logpai/loghub/tree

#!/usr/bin/env python3
"""
LogHub OpenStack Dataset - LogBERT Anomaly Detection Pipeline
Complete pipeline for LogHub OpenStack dataset with proper preprocessing and visualization
"""

import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import logging
from pathlib import Path
import requests
import zipfile
from urllib.parse import urljoin

# For LogBERT preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import BertTokenizer, BertModel
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class LogHubDatasetLoader:
    """Loader for LogHub OpenStack dataset with automatic download and preprocessing"""
    
    def __init__(self, base_url: str = "https://github.com/logpai/loghub/raw/master/OpenStack/"):
        self.base_url = base_url
        self.dataset_files = {
            'raw_logs': 'openstack_abnormal.log',
            'templates': 'openstack_templates.csv', 
            'event_sequence': 'openstack_sequence.csv',
            'anomaly_labels': 'openstack_abnormal_label.csv'
        }
        
    def download_dataset(self, output_dir: str = "loghub_openstack"):
        """Download LogHub OpenStack dataset files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Downloading LogHub OpenStack dataset...")
        
        # Alternative URLs and file names that might exist
        possible_files = [
            ('openstack_abnormal.log', 'raw_logs'),
            ('openstack_normal.log', 'normal_logs'),
            ('openstack_templates.csv', 'templates'),
            ('openstack_sequence.csv', 'sequences'),
            ('openstack_abnormal_label.csv', 'labels'),
            ('openstack.npz', 'processed_data')
        ]
        
        downloaded_files = {}
        
        for filename, file_type in possible_files:
            try:
                url = urljoin(self.base_url, filename)
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    file_path = output_path / filename
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    downloaded_files[file_type] = str(file_path)
                    print(f"✓ Downloaded {filename}")
                else:
                    print(f"✗ Could not download {filename} (status: {response.status_code})")
                    
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")
        
        return downloaded_files
    
    def create_sample_dataset(self, output_dir: str = "loghub_openstack"):
        """Create sample LogHub-style OpenStack dataset for demonstration"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Creating sample LogHub OpenStack dataset...")
        
        # Sample raw logs (typical OpenStack format from LogHub)
        sample_logs = [
            "2014-05-09 16:51:36.123 21691 INFO nova.compute.resource_tracker [req-12345 admin admin] Compute_service record updated for nova-compute",
            "2014-05-09 16:51:37.234 21691 ERROR nova.compute.manager [req-12346 user1 project1] Instance spawn failed",
            "2014-05-09 16:51:38.345 21691 INFO keystone.middleware.auth_token [req-12347] Authenticating user token", 
            "2014-05-09 16:51:39.456 21691 WARNING neutron.agent.dhcp.agent [req-12348] DHCP agent unable to process request",
            "2014-05-09 16:51:40.567 21691 DEBUG cinder.volume.drivers.lvm [req-12349] Volume volume-123 created successfully",
            "2014-05-09 16:51:41.678 21691 ERROR nova.network.manager [req-12350] Network connectivity lost",
            "2014-05-09 16:51:42.789 21691 INFO nova.compute.manager [req-12351 admin admin] Instance terminated successfully",
            "2014-05-09 16:51:43.890 21691 CRITICAL keystone.auth.controllers [req-12352] Authentication service unavailable",
            "2014-05-09 16:51:44.901 21691 INFO neutron.plugins.ml2.drivers.agent [req-12353] Port binding completed",
            "2014-05-09 16:51:45.012 21691 ERROR cinder.volume.manager [req-12354] Volume attachment failed"
        ]
        
        # Create raw log file
        with open(output_path / "openstack_abnormal.log", 'w') as f:
            for i in range(1000):  # Create larger dataset
                for log in sample_logs:
                    # Vary timestamps and IDs
                    varied_log = log.replace("16:51:36", f"16:51:{36+i%60}")
                    varied_log = varied_log.replace("req-12345", f"req-{12345+i}")
                    f.write(f"{varied_log}\n")
        
        # Create templates CSV (LogHub format)
        templates_data = {
            'EventTemplate': [
                'Compute_service record updated for *',
                'Instance spawn failed',
                'Authenticating user token',
                'DHCP agent unable to process request', 
                'Volume * created successfully',
                'Network connectivity lost',
                'Instance terminated successfully',
                'Authentication service unavailable',
                'Port binding completed',
                'Volume attachment failed'
            ],
            'EventId': [f'E{i:03d}' for i in range(10)]
        }
        pd.DataFrame(templates_data).to_csv(output_path / "openstack_templates.csv", index=False)
        
        # Create event sequences (LogHub format)
        sequences = []
        labels = []
        for i in range(200):  # 200 sequences
            seq_length = np.random.randint(5, 15)
            sequence = [f'E{np.random.randint(0, 10):03d}' for _ in range(seq_length)]
            sequences.append(' '.join(sequence))
            
            # Label as anomaly if contains error events (E001, E005, E007, E009)
            anomaly_events = {'E001', 'E005', 'E007', 'E009'}
            is_anomaly = any(event in anomaly_events for event in sequence)
            labels.append(1 if is_anomaly else 0)
        
        # Save sequences
        pd.DataFrame({
            'SessionId': [f'session_{i}' for i in range(len(sequences))],
            'EventSequence': sequences
        }).to_csv(output_path / "openstack_sequence.csv", index=False)
        
        # Save labels
        pd.DataFrame({
            'SessionId': [f'session_{i}' for i in range(len(labels))],
            'Label': labels
        }).to_csv(output_path / "openstack_abnormal_label.csv", index=False)
        
        print(f"✓ Created sample dataset in {output_path}/")
        return {
            'raw_logs': str(output_path / "openstack_abnormal.log"),
            'templates': str(output_path / "openstack_templates.csv"),
            'sequences': str(output_path / "openstack_sequence.csv"),
            'labels': str(output_path / "openstack_abnormal_label.csv")
        }

class LogHubOpenStackParser:
    """Parser specifically for LogHub OpenStack dataset format"""
    
    def __init__(self):
        self.log_pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+'
            r'(?P<pid>\d+)\s+'
            r'(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+'
            r'(?P<component>[\w\.]+)\s+'
            r'(?P<request_info>\[req-[^\]]+\])?\s*'
            r'(?P<message>.*)'
        )
        
    def parse_raw_logs(self, log_file_path: str) -> pd.DataFrame:
        """Parse raw OpenStack log file from LogHub"""
        print(f"Parsing raw logs from {log_file_path}...")
        
        logs = []
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                match = self.log_pattern.match(line)
                if match:
                    log_entry = match.groupdict()
                    log_entry['line_number'] = line_num
                    log_entry['raw_log'] = line
                    
                    # Extract request ID if present
                    if log_entry['request_info']:
                        req_match = re.search(r'req-([^\s\]]+)', log_entry['request_info'])
                        log_entry['request_id'] = req_match.group(1) if req_match else None
                    else:
                        log_entry['request_id'] = None
                        
                    logs.append(log_entry)
                else:
                    # Handle malformed lines
                    logs.append({
                        'line_number': line_num,
                        'raw_log': line,
                        'timestamp': None,
                        'pid': None,
                        'level': 'UNKNOWN',
                        'component': 'UNKNOWN',
                        'request_info': None,
                        'message': line,
                        'request_id': None
                    })
        
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        print(f"✓ Parsed {len(df)} log entries")
        return df
    
    def load_templates(self, templates_file_path: str) -> pd.DataFrame:
        """Load event templates from LogHub format"""
        print(f"Loading templates from {templates_file_path}...")
        
        templates_df = pd.read_csv(templates_file_path)
        print(f"✓ Loaded {len(templates_df)} event templates")
        return templates_df
    
    def load_sequences(self, sequences_file_path: str) -> pd.DataFrame:
        """Load event sequences from LogHub format"""
        print(f"Loading sequences from {sequences_file_path}...")
        
        sequences_df = pd.read_csv(sequences_file_path)
        print(f"✓ Loaded {len(sequences_df)} event sequences")
        return sequences_df
    
    def load_labels(self, labels_file_path: str) -> pd.DataFrame:
        """Load anomaly labels from LogHub format"""
        print(f"Loading labels from {labels_file_path}...")
        
        labels_df = pd.read_csv(labels_file_path)
        anomaly_count = labels_df['Label'].sum() if 'Label' in labels_df.columns else 0
        print(f"✓ Loaded {len(labels_df)} labels ({anomaly_count} anomalies)")
        return labels_df

class LogHubLogBERTPreprocessor:
    """Preprocessor for LogHub data to LogBERT format"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = 512
        
    def prepare_sequences_for_logbert(self, sequences_df: pd.DataFrame, 
                                    templates_df: pd.DataFrame,
                                    labels_df: pd.DataFrame) -> Dict:
        """Convert LogHub sequences to LogBERT input format"""
        
        print("Preparing sequences for LogBERT...")
        
        # Create template mapping
        template_map = dict(zip(templates_df['EventId'], templates_df['EventTemplate']))
        
        # Process sequences
        processed_sequences = []
        sequence_labels = []
        session_ids = []
        
        for _, row in sequences_df.iterrows():
            session_id = row['SessionId']
            event_sequence = row['EventSequence'].split()
            
            # Convert event IDs to templates
            template_sequence = []
            for event_id in event_sequence:
                if event_id in template_map:
                    template_sequence.append(template_map[event_id])
                else:
                    template_sequence.append(f"UNKNOWN_EVENT_{event_id}")
            
            # Join with [SEP] tokens for BERT
            combined_sequence = ' [SEP] '.join(template_sequence)
            processed_sequences.append(combined_sequence)
            session_ids.append(session_id)
            
            # Get label
            label_row = labels_df[labels_df['SessionId'] == session_id]
            label = label_row['Label'].iloc[0] if not label_row.empty else 0
            sequence_labels.append(label)
        
        # Tokenize sequences
        print("Tokenizing sequences...")
        encoded = self.tokenizer(
            processed_sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': torch.tensor(sequence_labels, dtype=torch.long),
            'session_ids': session_ids,
            'raw_sequences': processed_sequences,
            'event_sequences': [seq.split() for seq in sequences_df['EventSequence']]
        }
        
        print(f"✓ Prepared {len(processed_sequences)} sequences for LogBERT")
        print(f"  - Normal sequences: {(torch.tensor(sequence_labels) == 0).sum().item()}")
        print(f"  - Anomalous sequences: {(torch.tensor(sequence_labels) == 1).sum().item()}")
        
        return result
    
    def create_train_test_split(self, logbert_data: Dict, test_size: float = 0.2) -> Tuple[Dict, Dict]:
        """Split data for training and testing"""
        
        # Get indices for train/test split
        n_samples = len(logbert_data['session_ids'])
        indices = np.arange(n_samples)
        
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            stratify=logbert_data['labels'].numpy(),
            random_state=42
        )
        
        train_data = {
            'input_ids': logbert_data['input_ids'][train_idx],
            'attention_mask': logbert_data['attention_mask'][train_idx],
            'labels': logbert_data['labels'][train_idx],
            'session_ids': [logbert_data['session_ids'][i] for i in train_idx],
            'raw_sequences': [logbert_data['raw_sequences'][i] for i in train_idx],
            'event_sequences': [logbert_data['event_sequences'][i] for i in train_idx]
        }
        
        test_data = {
            'input_ids': logbert_data['input_ids'][test_idx],
            'attention_mask': logbert_data['attention_mask'][test_idx],
            'labels': logbert_data['labels'][test_idx], 
            'session_ids': [logbert_data['session_ids'][i] for i in test_idx],
            'raw_sequences': [logbert_data['raw_sequences'][i] for i in test_idx],
            'event_sequences': [logbert_data['event_sequences'][i] for i in test_idx]
        }
        
        print(f"✓ Train set: {len(train_data['session_ids'])} sequences")
        print(f"✓ Test set: {len(test_data['session_ids'])} sequences")
        
        return train_data, test_data

class LogHubAnomalyDetector:
    """Enhanced anomaly detector for LogHub OpenStack data"""
    
    def __init__(self):
        self.anomaly_patterns = {
            'error_events': {'E001', 'E005', 'E007', 'E009'},  # Events typically associated with errors
            'critical_sequences': [
                ['E001', 'E005'],  # spawn failure followed by network loss
                ['E007', 'E009'],  # auth failure followed by volume failure
            ],
            'rare_event_threshold': 0.05  # Events occurring less than 5% are considered rare
        }
        self.event_frequencies = {}
        
    def fit(self, train_data: Dict, templates_df: pd.DataFrame):
        """Train the detector on normal sequences"""
        print("Training anomaly detector...")
        
        # Calculate event frequencies
        all_events = []
        for sequence in train_data['event_sequences']:
            all_events.extend(sequence)
        
        event_counts = Counter(all_events)
        total_events = len(all_events)
        
        self.event_frequencies = {
            event: count / total_events 
            for event, count in event_counts.items()
        }
        
        print(f"✓ Calculated frequencies for {len(self.event_frequencies)} event types")
        
    def predict(self, test_data: Dict) -> np.ndarray:
        """Predict anomalies in test sequences"""
        print("Predicting anomalies...")
        
        predictions = []
        
        for sequence in test_data['event_sequences']:
            is_anomaly = self._is_anomalous_sequence(sequence)
            predictions.append(1 if is_anomaly else 0)
        
        predictions = np.array(predictions)
        anomaly_count = predictions.sum()
        
        print(f"✓ Detected {anomaly_count} anomalies out of {len(predictions)} sequences")
        
        return predictions
    
    def _is_anomalous_sequence(self, sequence: List[str]) -> bool:
        """Check if a single sequence is anomalous"""
        
        # Check for error events
        if any(event in self.anomaly_patterns['error_events'] for event in sequence):
            return True
        
        # Check for critical sequence patterns
        for critical_pattern in self.anomaly_patterns['critical_sequences']:
            if self._contains_pattern(sequence, critical_pattern):
                return True
        
        # Check for rare events
        rare_events = [
            event for event in sequence 
            if self.event_frequencies.get(event, 0) < self.anomaly_patterns['rare_event_threshold']
        ]
        
        if len(rare_events) > len(sequence) * 0.3:  # More than 30% rare events
            return True
        
        return False
    
    def _contains_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence contains a specific pattern"""
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        return False

class LogHubAnomalyVisualizer:
    """Advanced visualizer for LogHub OpenStack anomalies"""
    
    def __init__(self):
        self.colors = {
            'normal': '#2E8B57',      # Sea Green
            'anomaly': '#DC143C',     # Crimson
            'mixed': '#FF8C00'        # Dark Orange
        }
    
    def create_comprehensive_dashboard(self, test_data: Dict, predictions: np.ndarray, 
                                    templates_df: pd.DataFrame, output_dir: str):
        """Create comprehensive anomaly analysis dashboard"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Creating comprehensive dashboard...")
        
        # Main dashboard with multiple views
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Anomaly Detection Results', 'Event Distribution in Anomalies',
                'Sequence Length vs Anomaly', 'Template Usage Frequency', 
                'Anomaly Detection Timeline', 'Confusion Matrix'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.08
        )
        
        # 1. Overall results
        normal_count = (predictions == 0).sum()
        anomaly_count = (predictions == 1).sum()
        
        fig.add_trace(
            go.Bar(
                x=['Normal', 'Anomaly'],
                y=[normal_count, anomaly_count],
                marker_color=[self.colors['normal'], self.colors['anomaly']],
                name='Detection Results'
            ),
            row=1, col=1
        )
        
        # 2. Event distribution in anomalies
        anomaly_sequences = [test_data['event_sequences'][i] for i, pred in enumerate(predictions) if pred == 1]
        anomaly_events = []
        for seq in anomaly_sequences:
            anomaly_events.extend(seq)
        
        event_counts = Counter(anomaly_events)
        top_events = event_counts.most_common(10)
        
        if top_events:
            fig.add_trace(
                go.Bar(
                    x=[event for event, count in top_events],
                    y=[count for event, count in top_events],
                    marker_color=self.colors['anomaly'],
                    name='Anomaly Events'  
                ),
                row=1, col=2
            )
        
        # 3. Sequence length analysis
        sequence_lengths = [len(seq) for seq in test_data['event_sequences']]
        
        fig.add_trace(
            go.Scatter(
                x=sequence_lengths,
                y=predictions,
                mode='markers',
                marker=dict(
                    color=[self.colors['anomaly'] if pred else self.colors['normal'] for pred in predictions],
                    size=8,
                    opacity=0.6
                ),
                name='Length vs Anomaly'
            ),
            row=2, col=1
        )
        
        # 4. Template usage frequency
        template_map = dict(zip(templates_df['EventId'], templates_df['EventTemplate']))
        all_events = []
        for seq in test_data['event_sequences']:
            all_events.extend(seq)
        
        template_counts = Counter(all_events)
        top_templates = template_counts.most_common(8)
        
        if top_templates:
            template_names = [template_map.get(event, event) for event, count in top_templates]
            template_names = [name[:30] + '...' if len(name) > 30 else name for name in template_names]
            
            fig.add_trace(
                go.Bar(
                    x=template_names,
                    y=[count for event, count in top_templates],
                    marker_color=self.colors['mixed'],
                    name='Template Usage'
                ),
                row=2, col=2
            )
        
        # 5. Timeline (using sequence index as pseudo-time)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(predictions))),
                y=predictions,
                mode='markers',
                marker=dict(
                    color=[self.colors['anomaly'] if pred else self.colors['normal'] for pred in predictions],
                    size=6
                ),
                name='Detection Timeline'
            ),
            row=3, col=1
        )
        
        # 6. Confusion matrix (if ground truth available)
        if 'labels' in test_data:
            true_labels = test_data['labels'].numpy()
            cm = confusion_matrix(true_labels, predictions)
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Predicted Normal', 'Predicted Anomaly'],
                    y=['True Normal', 'True Anomaly'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            height=1200,
            title_text="LogHub OpenStack Anomaly Detection Dashboard",
            showlegend=False
        )
        
        # Save dashboard
        dashboard_path = output_path / "anomaly_dashboard.html"
        fig.write_html(dashboard_path)
        print(f"✓ Dashboard saved to {dashboard_path}")
        
        return fig
    
    def create_sequence_analysis(self, test_data: Dict, predictions: np.ndarray, 
                               templates_df: pd.DataFrame, output_dir: str):
        """Create detailed sequence analysis"""
        
        output_path = Path(output_dir)
        template_map = dict(zip(templates_df['EventId'], templates_df['EventTemplate']))
        
        # Detailed sequence analysis
        analysis_data = []
        
        for i, (session_id, sequence, prediction) in enumerate(zip(
            test_data['session_ids'], test_data['event_sequences'], predictions
        )):
            
            # Convert event IDs to readable templates
            readable_sequence = [template_map.get(event, event) for event in sequence]
            
            analysis_data.append({
                'session_id': session_id,
                'sequence_length': len(sequence),
                'event_sequence': ' -> '.join(sequence),
                'readable_sequence': ' -> '.join([tmpl[:20] + '...' if len(tmpl) > 20 else tmpl 
                                                for tmpl in readable_sequence]),
                'prediction': 'Anomaly' if prediction else 'Normal',
                'unique_events': len(set(sequence)),
                'has_error_events': any(event in ['E001', 'E005', 'E007', 'E009'] for event in sequence)
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # Save detailed analysis
        analysis_path = output_path / "sequence_analysis.csv"
        analysis_df.to_csv(analysis_path, index=False)
        print(f"✓ Sequence analysis saved to {analysis_path}")
        
        return analysis_df
    
    def generate_anomaly_report(self, test_data: Dict, predictions: np.ndarray, 
                              templates_df: pd.DataFrame, output_dir: str) -> str:
        """Generate comprehensive anomaly report"""
        
        output_path = Path(output_dir)
        true_labels = test_data.get('labels', None)
        
        report = f"""
LOGHUB OPENSTACK ANOMALY DETECTION REPORT
==========================================
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: LogHub OpenStack

DETECTION SUMMARY:
-----------------
Total Sequences Analyzed: {len(predictions)}
Anomalies Detected: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)
Normal Sequences: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(predictions)*100:.2f}%)

"""
        
        # Performance metrics if ground truth available
        if true_labels is not None:
            true_labels_np = true_labels.numpy()
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels_np, predictions)
            precision = precision_score(true_labels_np, predictions, zero_division=0)
            recall = recall_score(true_labels_np, predictions, zero_division=0)
            f1 = f1_score(true_labels_np, predictions, zero_division=0)
            
            report += f"""PERFORMANCE METRICS:
-------------------
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}

"""
        
        # Anomalous sequence analysis
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) > 0:
            report += "TOP ANOMALOUS SEQUENCES:\n"
            report += "-" * 24 + "\n"
            
            template_map = dict(zip(templates_df['EventId'], templates_df['EventTemplate']))
            
            for i, idx in enumerate(anomaly_indices[:10]):  # Top 10
                session_id = test_data['session_ids'][idx]
                sequence = test_data['event_sequences'][idx]
                readable_seq = [template_map.get(event, event)[:30] for event in sequence]
                
                report += f"{i+1}. Session: {session_id}\n"
                report += f"   Events: {' -> '.join(sequence)}\n"
                report += f"   Templates: {' -> '.join(readable_seq)}\n\n"
        
        # Event frequency analysis
        all_events = []
        anomaly_events = []
        
        for i, sequence in enumerate(test_data['event_sequences']):
            all_events.extend(sequence)
            if predictions[i] == 1:
                anomaly_events.extend(sequence)
        
        total_event_counts = Counter(all_events)
        anomaly_event
