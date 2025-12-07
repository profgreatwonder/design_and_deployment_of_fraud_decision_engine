# from datetime import datetime, time, timedelta, timezone
# import json
# import os
# import logging
# import random
# import signal
# from typing import Any, Dict, Optional

# from confluent_kafka import Producer
# from dotenv import load_dotenv
# from faker import Faker
# from jsonschema import FormatChecker, ValidationError, validate



# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
#     level=logging.INFO
# )

# logger = logging.getLogger(__name__)

# load_dotenv(dotenv_path="/app/.env")

# fake = Faker()

# TRANSACTION_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "transaction_id": {"type": "string"},
#         "user_id": {"type": "number", "minimum": 1000, "maximum": 9999},
#         "amount": {"type": "number", "minimum": 0.01, "maximum": 100000},
#         "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
#         "merchant": {"type": "string"},
#         "timestamp": {
#             "type": "string",
#             "format": "date-time"
#         },
#         "location": {"type": "string", "pattern": "^[A-Z]{2}$"},
#         "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1},
#         "mcc": {"type": "integer", "minimum": 1000, "maximum": 9999},
#         "card_type": {"type": "string"},
#         "card_brand": {"type": "string"},
#         "credit_score": {"type": "number", "minimum": 300, "maximum": 850}
#     },
#     "required": ["transaction_id", "user_id", "amount", "currency", "timestamp", "is_fraud", "mcc", "card_type", "card_brand", "credit_score"]
# }

# class TransactionProducer():
#     def __init__(self):
#         self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVER', 'localhost:9092')
#         # self.kafka_username = os.getenv('KAFKA_USERNAME')
#         # self.kafka_password = os.getenv('KAFKA_PASSWORD')
#         self.topic = os.getenv('KAFKA_TOPIC', 'transactions')
#         self.running = False





#         # # aiven kafka config
#         # self.producer_config = {
#         #     'bootstrap.servers': self.bootstrap_servers,
#         #     'client.id': 'transaction-producer',
#         #     'compression.type': 'gzip',
#         #     'linger.ms': '5',
#         #     'batch.size': 16384,
#         # }

#         # if self.kafka_username and self.kafka_password:
#         #     self.producer_config.update({
#         #         'bootstrap.servers': self.bootstrap_servers,
#         #         'security.protocol': 'SSL',
#         #         # 'security.protocol': 'SASL_SSL',
#         #         # 'sasl.mechanism': 'PLAIN',
#         #         # 'sasl.mechanism': 'SCRAM-SHA-512',
#         #         # 'sasl.username': self.kafka_username,
#         #         # 'sasl.password': self.kafka_password,

#         #         # # mTLS — THIS IS REQUIRED BY AIVEN
#         #         'ssl.ca.location': '/etc/ssl/certs/aiven-ca.pem',
#         #         'ssl.certificate.location': '/etc/ssl/certs/service.cert',
#         #         'ssl.key.location': '/etc/ssl/private/service.key',

#         #         # 'ssl.ca.location': '/app/ca.pem',  
#         #         # 'enable.ssl.certificate.verification': False
#         #     })

#         # else:
#         #     self.producer_config['security.protocol'] = 'PLAINTEXT'

#         # try:
#         #     self.producer = Producer(self.producer_config)
#         #     # logger.info('Confluent kafka producer is initialized successfully')
#         #     logger.info('Aiven kafka producer is initialized successfully')

#         # except Exception as e:
#         #     # logger.error(f'Failed to initialize confluent kafka producer: {str(e)}')
#         #     logger.error(f'Failed to initialize aiven kafka producer: {str(e)}')
#         #     raise e


#         # Local Kafka config (no SSL required)
#         self.producer_config = {
#             'bootstrap.servers': self.bootstrap_servers,
#             'client.id': 'transaction-producer',
#             'compression.type': 'gzip',
#             'linger.ms': '5',
#             'batch.size': 16384,
#             'security.protocol': 'PLAINTEXT'
#         }

#         try:
#             self.producer = Producer(self.producer_config)
#             logger.info('Local Kafka producer initialized successfully')
#         except Exception as e:
#             logger.error(f'Failed to initialize Kafka producer: {str(e)}')
#             raise e        
        
#         # self.compromised_users = set(random.sample(range(1000, 9999), 50)) # make sure that at least 0.5% of the users are fraudulent
#         # self.high_risk_merchants = ['QuickCash', 'GlobslDigital', 'FastMoneyX']
#         # self.fraud_pattern_weights = {
#         #     'account_takeover': 0.4, #40% of fraud cases
#         #     'card_testing': 0.3, #30% of fraud causes
#         #     'merchant_collusion': 0.2, #20%
#         #     'geo_anomaly': 0.1 #10%

#         # }


#         self.compromised_users = set(random.sample(range(1000, 9999), 50))
#         self.high_risk_merchants = ['QuickCash', 'GlobalDigital', 'FastMoneyX']
#         self.card_types = ['credit', 'debit', 'prepaid']
#         self.card_brands = ['Visa', 'MasterCard', 'Amex', 'Discover']
#         self.high_risk_mccs = [4411, 5733, 3005, 5045, 5732, 5533, 5816, 3144, 4131, 5712]

#         # configure graceful shutdown
#         signal.signal(signal.SIGINT, self.shutdown)
#         signal.signal(signal.SIGTERM, self.shutdown)


#     def delivery_report(self, err, msg):
#         if err is not None:
#             logger.error(f'Message delivery failed: {err}')
#         else:
#             logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')


#     def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
#         try:
#             validate(
#                 instance=transaction,
#                 schema=TRANSACTION_SCHEMA,
#                 format_checker=FormatChecker()
#             )
#         except ValidationError as e:
#             logger.error(f'Invalid transaction: {e.message}')
            
#     def generate_transaction(self) -> Optional[Dict[str, Any]]:
#         transaction = {
#             'transaction_id': fake.uuid4(),
#             'user_id': random.randint(1000, 9999),
#             'amount': round(fake.pyfloat(min_value=0.01, max_value=10000), 2),
#             'currency': 'USD',
#             'merchant': fake.company(),
#             'timestamp': (datetime.now(timezone.utc) + timedelta(seconds=random.randint(-300, 3000))).isoformat(),
#             'location': fake.country_code(),
#             'mcc': random.randint(1000, 9999),
#             'card_type': random.choice(self.card_types),
#             'card_brand': random.choice(self.card_brands),
#             'credit_score': random.randint(300, 850),
#             'is_fraud': 0
#         }

#         is_fraud = 0
#         amount = transaction['amount']
#         user_id = transaction['user_id']
#         merchant = transaction['merchant']
#         mcc = transaction['mcc']
        
#         # account takeover
#         if user_id in self.compromised_users and amount > 500:
#             if random.random() < 0.3: #30% change of fraud in compromised accounts
#                 is_fraud = 1
#                 transaction['amount'] = random.uniform(500, 5000)
#                 transaction['merchant'] = random.choice(self.high_risk_merchants)


#         # card testing
#         if is_fraud and amount < 2.0:
#             # simulate rapid small transactions
#             if user_id % 1000 == 0 and random.random() < 0.25:
#                 is_fraud = 1
#                 transaction['amount'] = round(random.uniform(0.01, 2), 2)
#                 transaction['location'] = 'US'

#         # merchant collusion
#         if not is_fraud and merchant in self.high_risk_merchants:
#             if amount > 3000 and random.random() < 0.15:
#                 is_fraud = 1
#                 transaction['amount'] = round.uniform(300, 1500)

#         # geographic anomaly
#         if not is_fraud:
#             if user_id % 500 == 0 and random.random() < 0.1:
#                 is_fraud = 1
#                 # transaction['location'] = random.choice(['CN', 'RU', 'GB'])
#                 transaction['location'] = random.choice(['CN', 'RU', 'GB', 'NG', 'BR'])

#         # baseline for random fraud (0.1, 0.3%)
#         if not is_fraud and random.random() < 0.002:
#             is_fraud = 1
#             transaction['amount'] = random.uniform(100, 2000)

#         # ensure that the final fraud rate is between 1-2%
#         transaction['is_fraud'] = is_fraud if random.random() < 0.985 else 0

#         # validate modified transaction
#         if self.validate_transaction(transaction):
#             return transaction

#     def send_transaction(self) -> bool:
#         try:
#             transaction = self.generate_transaction()
#             if not transaction:
#                 return False

#             self.producer.produce(
#                 self.topic,
#                 key=transaction['transaction_id'],
#                 value=json.dumps(transaction),
#                 callback=self.delivery_report
#             )

#             self.producer.poll(0) #trigger callbacks
#             return True

#         except Exception as e:
#             logger.error(f'Error producing message: {str(e)}')
#             return False


#     def run_continuous_production(self, interval: float=0.0):
#         # run continuous message production with graceful shutdown
#         self.running = True
#         logger.info('Starting producer for topic %s...', self.topic)

#         try:
#             while self.running:
#                 if self.send_transaction():
#                     time.sleep(interval)

#         finally:
#             self.shutdown()


#     def shutdown(self, signum=None, frame=None):
#         if self.running:
#             logger.info('Initializing shutdown...')
#             self.running = False

#             if self.producer:
#                 self.producer.flush(timeout=30)
#                 self.producer.close()
#             logger.info('Producer stopped')


# if __name__ == "__main__":
#     producer = TransactionProducer()
#     producer.run_continuous_production()



"""
Transaction Producer for Local Kafka
Reads from data/folder and sends transactions to Kafka
"""

import sys
import os

from datetime import datetime, timedelta, timezone
import json
import logging
import random
import signal
import time
from typing import Any, Dict, Optional
import yaml

# Import with version check
import numpy as np
np_version = np.__version__
logging.info(f"NumPy version: {np_version}")

import pandas as pd
pd_version = pd.__version__
logging.info(f"Pandas version: {pd_version}")

from confluent_kafka import Producer
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv()


class TransactionProducer:
    def __init__(self, config_path=None):
    # def __init__(self):
        logger.info("🚀 Initializing Transaction Producer...")
        
        if config_path is None:
            # Get the directory where THIS script (main.py) lives
            # Docker path: /app/src/producer
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Go up one level to 'src', then down to 'config'
            # Result: /app/src/config/config.yaml
            project_src_root = os.path.dirname(current_script_dir)
            config_path = os.path.join(project_src_root, 'config', 'config.yaml')

        logger.info(f"Loading config from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config file not found at: {config_path}")

        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("✅ Configuration loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {str(e)}")
            raise
        
        self.kafka_config = self.config['kafka']
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVER', self.kafka_config['bootstrap_servers'])
        self.topic = os.getenv('KAFKA_TOPIC', self.kafka_config['topic'])
        self.running = False
        
        # Local Kafka config (no SSL)
        self.producer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': 'transaction-producer',
            'compression.type': 'gzip',
            'linger.ms': 5,
            'batch.size': 16384,
        }
        
        try:
            self.producer = Producer(self.producer_config)
            logger.info('✅ Local Kafka producer initialized successfully')
        except Exception as e:
            logger.error(f'❌ Failed to initialize Kafka producer: {str(e)}')
            raise e
        
        # Load transaction data
        self.load_transactions()
        
        # Configure graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def load_transactions(self):
        """Load transactions from merged dataset"""
        try:
            # Try to load preprocessed data
            data_path = self.config['data']['merged_output']
            if os.path.exists(data_path):
                logger.info(f"📂 Loading from {data_path}...")
                self.transactions_df = pd.read_csv(data_path)
                logger.info(f"✅ Loaded {len(self.transactions_df)} transactions from merged dataset")
            else:
                # Load raw transaction data
                data_path = self.config['data']['transactions_data']
                logger.info(f"📂 Loading from {data_path}...")
                self.transactions_df = pd.read_csv(data_path)
                logger.info(f"✅ Loaded {len(self.transactions_df)} raw transactions")
            
            # Clean column names
            self.transactions_df.columns = self.transactions_df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            self.transaction_index = 0
            logger.info(f"📊 Available columns: {list(self.transactions_df.columns)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load transactions: {str(e)}")
            self.transactions_df = pd.DataFrame()
            self.transaction_index = 0
    
    def delivery_report(self, err, msg):
        """Kafka delivery callback"""
        if err is not None:
            logger.error(f'❌ Message delivery failed: {err}')
        else:
            logger.info(f'✅ Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def get_next_transaction(self) -> Optional[Dict[str, Any]]:
        """Get next transaction from dataset"""
        if self.transactions_df.empty:
            logger.warning("⚠️  No transactions available")
            return None
        
        if self.transaction_index >= len(self.transactions_df):
            logger.info("🔄 Reached end of transactions, restarting from beginning")
            self.transaction_index = 0
        
        row = self.transactions_df.iloc[self.transaction_index]
        self.transaction_index += 1
        
        # Convert row to transaction dict with safe type conversion
        def safe_get(row, key, default, converter=None):
            """Safely get value from row with type conversion"""
            try:
                val = row.get(key, default)
                if pd.isna(val):
                    return default
                return converter(val) if converter else val
            except:
                return default
        
        transaction = {
            'transaction_id': safe_get(row, 'id', f'txn_{self.transaction_index}', str),
            'user_id': safe_get(row, 'client_id', 0, int),
            'amount': safe_get(row, 'amount', 0.0, float),
            'currency': safe_get(row, 'currency', 'USD', str),
            'merchant': safe_get(row, 'merchant_id', 'Unknown', str),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'location': safe_get(row, 'merchant_state', 'US', str),
            'mcc': safe_get(row, 'mcc', 0, int),
        }
        
        # Add optional fields if available
        optional_fields = {
            'card_brand': str,
            'card_type': str,
            'use_chip': str,
            'has_chip': str,
            'gender': str,
            'card_on_dark_web': str,
            'credit_limit': float,
            'merchant_city': str,
            'description': str
        }
        
        for field, converter in optional_fields.items():
            val = safe_get(row, field, None, converter)
            if val is not None:
                transaction[field] = val
        
        return transaction
    
    def send_transaction(self) -> bool:
        """Send transaction to Kafka"""
        try:
            transaction = self.get_next_transaction()
            if not transaction:
                return False
            
            logger.info(f"📤 Sending transaction {transaction['transaction_id']}: ${transaction['amount']:.2f}")
            
            self.producer.produce(
                self.topic,
                key=str(transaction['transaction_id']),
                value=json.dumps(transaction),
                callback=self.delivery_report
            )
            
            self.producer.poll(0)
            return True
            
        except Exception as e:
            logger.error(f'❌ Error producing message: {str(e)}')
            return False
    
    def run_continuous_production(self, interval: float = 1.0):
        """Run continuous message production"""
        self.running = True
        logger.info(f'🚀 Starting producer for topic {self.topic}...')
        logger.info(f'⏱️  Sending 1 transaction every {interval} seconds')
        
        transaction_count = 0
        
        try:
            while self.running:
                if self.send_transaction():
                    transaction_count += 1
                    if transaction_count % 10 == 0:
                        logger.info(f"📊 Sent {transaction_count} transactions so far...")
                    time.sleep(interval)
                else:
                    logger.warning("⚠️  No transaction sent, waiting...")
                    time.sleep(5)
        finally:
            self.shutdown()
    
    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        if self.running:
            logger.info('🛑 Initializing shutdown...')
            self.running = False
            
            if self.producer:
                logger.info('⏳ Flushing remaining messages...')
                self.producer.flush(timeout=30)
            logger.info('✅ Producer stopped successfully')


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 FRAUD DETECTION SYSTEM - TRANSACTION PRODUCER")
    logger.info("="*60)
    
    producer = TransactionProducer()
    producer.run_continuous_production(interval=2.0)  # Send 1 transaction every 2 seconds