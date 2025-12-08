"""
Data Preprocessor for Fraud Detection System
Loads and processes data from the data/ folder
"""

import pandas as pd
import numpy as np
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path="/opt/airflow/config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.high_risk_mccs = self.config['high_risk_mccs']
        
    # def load_data(self):
    #     """Load all data files from data/ folder"""
    #     logger.info("Loading datasets from data folder...")
        
    #     # Load cards data
    #     self.cards_df = pd.read_csv(self.data_config['cards_data'])
    #     logger.info(f"Loaded cards_data: {len(self.cards_df)} rows")
        
    #     # Load MCC codes
    #     with open(self.data_config['mcc_codes'], 'r') as f:
    #         data = json.load(f)
    #         self.mcc_df = pd.DataFrame(list(data.items()), columns=['MCC', 'Description'])
    #     logger.info(f"Loaded mcc_codes: {len(self.mcc_df)} rows")
        
    #     # Load transactions
    #     self.transactions_df = pd.read_csv(self.data_config['transactions_data'])
    #     logger.info(f"Loaded transactions_data: {len(self.transactions_df)} rows")
        
    #     # Load users
    #     self.user_df = pd.read_csv(self.data_config['users_data'])
    #     logger.info(f"Loaded users_data: {len(self.user_df)} rows")
        
    #     # Load fraud labels
    #     with open(self.data_config['fraud_labels'], 'r') as f:
    #         data = json.load(f)
    #         if isinstance(data, dict) and 'target' in data:
    #             self.labels_df = pd.DataFrame.from_dict(data['target'], orient='index', columns=['target'])
    #         else:
    #             self.labels_df = pd.DataFrame.from_dict(data, orient='index', columns=['target'])
    #         self.labels_df.index.name = 'id'
    #     logger.info(f"Loaded fraud_labels: {len(self.labels_df)} rows")
        
    #     # Clean column names
    #     for df in [self.cards_df, self.mcc_df, self.transactions_df, self.user_df, self.labels_df]:
    #         df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
    #     return self
    

    def load_data(self):
        """Load all data files from data/ folder with memory optimization"""
        logger.info("Loading datasets from data folder...")
        
        # Define types to drastically reduce memory usage
        dtypes = {
            'client_id': 'int32',
            'card_id': 'int32',
            'amount': 'string', # We clean this later, so keep as string for now
            'mcc': 'int16',
            'use_chip': 'category',
            'card_brand': 'category',
            'card_type': 'category',
            'is_fraud': 'int8'
        }

        try:
            # Load smaller files normally
            self.cards_df = pd.read_csv(self.data_config['cards_data'])
            
            with open(self.data_config['mcc_codes'], 'r') as f:
                data = json.load(f)
                self.mcc_df = pd.DataFrame(list(data.items()), columns=['MCC', 'Description'])
            
            # Load the HUGE file with optimization
            logger.info("Loading transactions_data (optimized)...")
            self.transactions_df = pd.read_csv(
                self.data_config['transactions_data'], 
                dtype=dtypes,
                parse_dates=['date']  # Parse dates while reading to save a step
            )
            logger.info(f"Loaded transactions_data: {len(self.transactions_df)} rows")
            
            # Load users
            self.user_df = pd.read_csv(self.data_config['users_data'])
            
            # Load labels
            with open(self.data_config['fraud_labels'], 'r') as f:
                data = json.load(f)
                # Handle dictionary structure variations
                if isinstance(data, dict) and 'target' in data:
                    self.labels_df = pd.DataFrame.from_dict(data['target'], orient='index', columns=['target'])
                else:
                    self.labels_df = pd.DataFrame.from_dict(data, orient='index', columns=['target'])
                self.labels_df.index.name = 'id'
            
            # Clean column names
            for df in [self.cards_df, self.mcc_df, self.transactions_df, self.user_df, self.labels_df]:
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            return self

        except MemoryError:
            logger.error("❌ OOM Error: Dataset too large for Docker RAM limits.")
            raise
    def merge_data(self):
        """Merge all datasets"""
        logger.info("Merging datasets...")
        
        # Start with transactions
        df = self.transactions_df.copy()
        
        # Add fraud labels
        labels_df_temp = self.labels_df.reset_index()
        labels_df_temp.rename(columns={'index': 'id', 'target': 'is_fraud'}, inplace=True)
        labels_df_temp['id'] = labels_df_temp['id'].astype('Int64')
        df['id'] = df['id'].astype('Int64')
        df = pd.merge(df, labels_df_temp, on='id', how='left')
        df['is_fraud'] = df['is_fraud'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Merge with cards
        cards_df_temp = self.cards_df.copy()
        cards_df_temp = cards_df_temp.rename(columns={'id': 'card_id'})
        cards_df_temp = cards_df_temp.drop(columns=['client_id'])
        cards_df_temp['card_id'] = cards_df_temp['card_id'].astype('Int64')
        df['card_id'] = df['card_id'].astype('Int64')
        df = pd.merge(df, cards_df_temp, on='card_id', how='left')
        
        # Merge with MCC
        self.mcc_df['mcc'] = self.mcc_df['mcc'].astype('Int64')
        df['mcc'] = df['mcc'].astype('Int64')
        df = pd.merge(df, self.mcc_df, on='mcc', how='left')
        
        # Merge with users
        user_df_temp = self.user_df.copy()
        user_df_temp = user_df_temp.rename(columns={'id': 'client_id'})
        user_df_temp['client_id'] = user_df_temp['client_id'].astype('Int64')
        df['client_id'] = df['client_id'].astype('Int64')
        df = pd.merge(df, user_df_temp, on='client_id', how='left')
        
        self.merged_df = df
        logger.info(f"Merged dataset: {len(self.merged_df)} rows, {len(self.merged_df.columns)} columns")
        
        return self
    
    def preprocess(self):
        """Apply all preprocessing steps"""
        logger.info("Preprocessing data...")
        df = self.merged_df
        
        # Handle missing values
        df['errors'].fillna('No', inplace=True)
        df['zip'].fillna('Unknown', inplace=True)
        df['merchant_state'].fillna('Unknown', inplace=True)
        df['cvv'] = df['cvv'].fillna(0).astype(int).astype(str).str.zfill(3)
        
        # Drop rows with critical missing values
        df.dropna(subset=['amount', 'use_chip', 'merchant_id', 'mcc', 'description', 'merchant_city'], inplace=True)
        
        # Convert currency columns
        currency_cols = ['amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('-', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Sort by client and date
        df = df.sort_values(by=['client_id', 'date']).reset_index(drop=True)
        
        self.merged_df = df
        logger.info("Basic preprocessing completed")
        
        return self
    
    def engineer_features(self):
        """Create engineered features"""
        logger.info("Engineering features...")
        df = self.merged_df
        
        # Credit utilization
        df['credit_utilization'] = np.where(
            df['credit_limit'] != 0,
            df['amount'] / df['credit_limit'],
            0
        )
        df['credit_utilization'].fillna(0, inplace=True)
        
        # Transaction frequency (past-only)
        df['temp_daily_group'] = df['date'].dt.date
        df['transactions_per_day_past'] = df.groupby(['client_id', 'temp_daily_group']).cumcount()
        
        df['temp_weekly_group'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
        df['transactions_per_week_past'] = df.groupby(['client_id', 'temp_weekly_group']).cumcount()
        
        df = df.drop(columns=['temp_daily_group', 'temp_weekly_group'])
        
        # Rolling statistics (3-day window, past only)
        rolling_mean_series = df.groupby('client_id').rolling(window='3D', on='date', closed='left')['amount'].mean()
        rolling_mean_df = rolling_mean_series.reset_index()
        rolling_mean_df.rename(columns={'amount': 'rolling_mean_3day'}, inplace=True)
        
        rolling_std_series = df.groupby('client_id').rolling(window='3D', on='date', closed='left')['amount'].std()
        rolling_std_df = rolling_std_series.reset_index()
        rolling_std_df.rename(columns={'amount': 'rolling_std_3day'}, inplace=True)
        
        df = pd.merge(df, rolling_mean_df[['client_id', 'date', 'rolling_mean_3day']], on=['client_id', 'date'], how='left')
        df = pd.merge(df, rolling_std_df[['client_id', 'date', 'rolling_std_3day']], on=['client_id', 'date'], how='left')
        
        df['rolling_mean_3day'] = df['rolling_mean_3day'].fillna(0)
        df['rolling_std_3day'] = df['rolling_std_3day'].fillna(0)
        
        # Age at account open
        df['acct_open_date'] = pd.to_datetime(df['acct_open_date'], format='%m/%Y', errors='coerce')
        df['acct_open_year'] = df['acct_open_date'].dt.year
        df['age_at_acct_open'] = df['acct_open_year'] - df['birth_year']
        df['age_at_acct_open'] = df['age_at_acct_open'].fillna(0)
        
        # Card age
        df['card_age'] = (df['date'] - df['acct_open_date']).dt.days / 365.25
        df['card_age'] = df['card_age'].fillna(0)
        
        # High-risk MCC flag
        df['is_high_risk_mcc'] = df['mcc'].isin(self.high_risk_mccs).astype(int)
        
        self.merged_df = df
        logger.info("Feature engineering completed")
        
        return self
    
    def get_processed_data(self):
        """Return processed dataframe"""
        return self.merged_df
    
    def save_processed_data(self):
        """Save processed data"""
        output_path = self.data_config['merged_output']
        self.merged_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        return self
    
    def process_all(self):
        """Run full preprocessing pipeline"""
        self.load_data()
        self.merge_data()
        self.preprocess()
        self.engineer_features()
        self.save_processed_data()
        
        logger.info(f"Final dataset shape: {self.merged_df.shape}")
        logger.info(f"Fraud rate: {self.merged_df['is_fraud'].mean():.4f}")
        
        return self.merged_df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.process_all()
    print(f"\nProcessing complete! Dataset shape: {df.shape}")
    print(f"Fraud transactions: {df['is_fraud'].sum()}")
    print(f"Non-fraud transactions: {(df['is_fraud'] == 0).sum()}")