import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_merge_data(self, file_paths: Dict) -> pd.DataFrame:
        """Load and merge multiple data sources"""
        logger.info("Loading and merging datasets...")
        
        # Load transactions
        transactions = pd.read_csv(file_paths['transactions'])
        
        # Load other datasets
        cards = pd.read_csv(file_paths['cards'])
        users = pd.read_csv(file_paths['users'])
        
        # Load MCC codes
        with open(file_paths['mcc'], 'r') as f:
            mcc_data = json.load(f)
        mcc_df = pd.DataFrame(list(mcc_data.items()), columns=['mcc', 'description'])
        
        # Load fraud labels
        with open(file_paths['labels'], 'r') as f:
            labels_data = json.load(f)
        labels_df = pd.DataFrame.from_dict(labels_data['target'], orient='index', columns=['is_fraud'])
        labels_df.index.name = 'id'
        
        # Merge datasets
        df = transactions.copy()
        df = pd.merge(df, labels_df, left_on='id', right_index=True, how='left')
        df = pd.merge(df, cards.rename(columns={'id': 'card_id'}), on='card_id', how='left')
        df = pd.merge(df, mcc_df, on='mcc', how='left')
        df = pd.merge(df, users.rename(columns={'id': 'client_id'}), on='client_id', how='left')
        
        logger.info(f"Merged dataset shape: {df.shape}")
        return df
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names"""
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type"""
        # Fill categorical columns
        cat_columns = ['errors', 'zip', 'merchant_state']
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Drop rows with critical missing values
        critical_columns = ['amount', 'use_chip', 'merchant_id', 'mcc', 'description', 'merchant_city']
        df = df.dropna(subset=critical_columns)
        
        # Fill numerical columns with median
        num_columns = df.select_dtypes(include=[np.number]).columns
        df[num_columns] = self.imputer.fit_transform(df[num_columns])
        
        return df
    
    def convert_currency_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert currency columns to numeric"""
        currency_columns = ['amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']
        
        for col in currency_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('-', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from date column"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Apply all preprocessing steps
        df = self.clean_column_names(df)
        df = self.handle_missing_values(df)
        df = self.convert_currency_columns(df)
        df = self.extract_date_features(df)
        
        logger.info(f"Preprocessed dataset shape: {df.shape}")
        return df