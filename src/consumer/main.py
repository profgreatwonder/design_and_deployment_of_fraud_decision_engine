"""
Transaction Consumer with Fraud Decisioning, Database Storage, and Alerts
"""

import sys
import os

# SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.insert(0, SRC_PATH)

# Add the parent directory of the current file's directory to sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from src.decisioning_engine.decision_engine import FraudDecisionEngine


import json
import logging
import signal
from typing import Dict, Any
from datetime import datetime
import yaml
import psycopg2
from psycopg2.extras import Json

from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

# from decisioning_engine.decision_engine import FraudDecisionEngine

# # Fix import paths - try multiple locations
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)

# # Add possible src locations to path
# for path in ['/app', '/app/src', parent_dir, grandparent_dir]:
#     if path not in sys.path:
#         sys.path.insert(0, path)

# Try to import decision engine from different locations
# try:
#     from decisioning_engine.decision_engine import FraudDecisionEngine
#     logger = logging.getLogger(__name__)
#     logger.info("✅ Imported from decisioning_engine.decision_engine")
# except ImportError:
#     try:
#         from src.decisioning_engine.decision_engine import FraudDecisionEngine
#         logger = logging.getLogger(__name__)
#         logger.info("✅ Imported from src.decisioning_engine.decision_engine")
#     except ImportError:
#         # Last resort - import from absolute path
#         sys.path.insert(0, '/app/src/decisioning_engine')
#         from decisioning_engine.decision_engine import FraudDecisionEngine
#         logger = logging.getLogger(__name__)
#         logger.info("✅ Imported from decision_engine (direct)")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv()


class FraudStatistics:
    """Track fraud detection statistics"""
    def __init__(self):
        self.total_transactions = 0
        self.decisions = {'ALLOW': 0, 'REJECT': 0, 'PEND': 0, 'ERROR': 0}
        self.total_amount_processed = 0.0
        self.total_amount_rejected = 0.0
        self.avg_latency = 0.0
        self.latencies = []
    
    def update(self, decision: str, amount: float, latency: float):
        self.total_transactions += 1
        self.decisions[decision] = self.decisions.get(decision, 0) + 1
        self.total_amount_processed += amount
        if decision == 'REJECT':
            self.total_amount_rejected += amount
        self.latencies.append(latency)
        self.avg_latency = sum(self.latencies) / len(self.latencies)
    
    def get_summary(self) -> Dict:
        return {
            'total_transactions': self.total_transactions,
            'decisions': self.decisions,
            'total_amount_processed': round(self.total_amount_processed, 2),
            'total_amount_rejected': round(self.total_amount_rejected, 2),
            'avg_latency_ms': round(self.avg_latency, 2),
            'fraud_detection_rate': round(
                (self.decisions.get('REJECT', 0) / max(self.total_transactions, 1)) * 100, 2
            )
        }


class DatabaseManager:
    """Manage database connections and operations"""
    def __init__(self):
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                database=os.getenv('POSTGRES_DB', 'fraud_detection'),
                user=os.getenv('POSTGRES_USER', 'airflow'),
                password=os.getenv('POSTGRES_PASSWORD', 'airflow')
            )
            self.conn.autocommit = True
            logger.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            self.conn = None
    
    def create_tables(self):
        """Create necessary tables"""
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Create decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_decisions (
                    id SERIAL PRIMARY KEY,
                    transaction_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id INTEGER,
                    amount DECIMAL(10, 2),
                    merchant VARCHAR(255),
                    mcc INTEGER,
                    fraud_probability DECIMAL(5, 4),
                    decision VARCHAR(10),
                    latency_ms DECIMAL(10, 2),
                    transaction_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_alerts (
                    id SERIAL PRIMARY KEY,
                    transaction_id VARCHAR(255) NOT NULL,
                    alert_type VARCHAR(20),
                    severity VARCHAR(10),
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transaction_id) REFERENCES fraud_decisions(transaction_id)
                )
            """)
            
            # Create statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_statistics (
                    id SERIAL PRIMARY KEY,
                    date DATE DEFAULT CURRENT_DATE,
                    total_transactions INTEGER,
                    allow_count INTEGER,
                    reject_count INTEGER,
                    pend_count INTEGER,
                    total_amount_processed DECIMAL(15, 2),
                    total_amount_rejected DECIMAL(15, 2),
                    avg_latency_ms DECIMAL(10, 2),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            cursor.close()
            logger.info("✅ Database tables created/verified")
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
    
    def save_decision(self, transaction: Dict, result: Dict):
        """Save fraud decision to database"""
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO fraud_decisions 
                (transaction_id, user_id, amount, merchant, mcc, fraud_probability, 
                 decision, latency_ms, transaction_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO UPDATE SET
                    fraud_probability = EXCLUDED.fraud_probability,
                    decision = EXCLUDED.decision,
                    latency_ms = EXCLUDED.latency_ms
            """, (
                result.get('transaction_id'),
                transaction.get('user_id'),
                transaction.get('amount'),
                transaction.get('merchant'),
                transaction.get('mcc'),
                result.get('fraud_probability'),
                result.get('decision'),
                result.get('latency_ms'),
                Json(transaction)
            ))
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error saving decision: {str(e)}")
    
    def save_alert(self, transaction_id: str, alert_type: str, severity: str, message: str):
        """Save fraud alert"""
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO fraud_alerts (transaction_id, alert_type, severity, message)
                VALUES (%s, %s, %s, %s)
            """, (transaction_id, alert_type, severity, message))
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error saving alert: {str(e)}")
    
    def update_statistics(self, stats: Dict):
        """Update daily statistics"""
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO fraud_statistics 
                (date, total_transactions, allow_count, reject_count, pend_count,
                 total_amount_processed, total_amount_rejected, avg_latency_ms)
                VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    total_transactions = EXCLUDED.total_transactions,
                    allow_count = EXCLUDED.allow_count,
                    reject_count = EXCLUDED.reject_count,
                    pend_count = EXCLUDED.pend_count,
                    total_amount_processed = EXCLUDED.total_amount_processed,
                    total_amount_rejected = EXCLUDED.total_amount_rejected,
                    avg_latency_ms = EXCLUDED.avg_latency_ms,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                stats['total_transactions'],
                stats['decisions']['ALLOW'],
                stats['decisions']['REJECT'],
                stats['decisions']['PEND'],
                stats['total_amount_processed'],
                stats['total_amount_rejected'],
                stats['avg_latency_ms']
            ))
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class TransactionConsumer:
    def __init__(self, config_path=None):
        # Find config file
        if config_path is None:
            possible_paths = [
                "/app/src/config/config.yaml",
                "/src/config/config.yaml",
                "src/config/config.yaml",
                "../config/config.yaml"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError(f"Config file not found. Tried: {possible_paths}")
        
        logger.info(f"Loading config from: {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kafka_config = self.config['kafka']
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVER', self.kafka_config['bootstrap_servers'])
        self.topic = os.getenv('KAFKA_TOPIC', self.kafka_config['topic'])
        self.group_id = os.getenv('KAFKA_CONSUMER_GROUP_ID', self.kafka_config['consumer_group_id'])
        self.running = False
        
        # Local Kafka consumer config (no SSL)
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'client.id': 'transaction-consumer',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
        }
        
        try:
            self.consumer = Consumer(self.consumer_config)
            self.consumer.subscribe([self.topic])
            logger.info(f'✅ Local Kafka consumer initialized, subscribed to: {self.topic}')
        except Exception as e:
            logger.error(f'❌ Failed to initialize Kafka consumer: {str(e)}')
            raise e
        
        # Initialize decision engine
        try:
            self.decision_engine = FraudDecisionEngine(config_path)
            logger.info('✅ Fraud decision engine initialized successfully')
        except Exception as e:
            logger.error(f'❌ Failed to initialize decision engine: {str(e)}')
            self.decision_engine = None
        
        # Initialize database manager
        self.db = DatabaseManager()
        
        # Initialize statistics
        self.stats = FraudStatistics()
        self.stats_update_counter = 0
        
        # Configure graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def send_alert(self, result: Dict, transaction: Dict):
        """Send alerts for REJECT and PEND decisions"""
        decision = result['decision']
        fraud_prob = result['fraud_probability']
        transaction_id = result['transaction_id']
        amount = transaction.get('amount', 0)
        
        if decision == 'REJECT':
            message = (
                f"🚨 FRAUD ALERT: Transaction {transaction_id} REJECTED\n"
                f"Amount: ${amount:.2f}\n"
                f"Fraud Probability: {fraud_prob:.2%}\n"
                f"User: {transaction.get('user_id')}\n"
                f"Merchant: {transaction.get('merchant')}"
            )
            logger.error(message)
            self.db.save_alert(transaction_id, 'REJECT', 'HIGH', message)
            
        elif decision == 'PEND':
            message = (
                f"⚠️  REVIEW REQUIRED: Transaction {transaction_id} PENDING\n"
                f"Amount: ${amount:.2f}\n"
                f"Fraud Probability: {fraud_prob:.2%}\n"
                f"User: {transaction.get('user_id')}\n"
                f"Merchant: {transaction.get('merchant')}"
            )
            logger.warning(message)
            self.db.save_alert(transaction_id, 'PEND', 'MEDIUM', message)
    
    def process_transaction(self, transaction: Dict[str, Any]) -> None:
        """Process transaction with fraud detection"""
        try:
            logger.info(f"📥 Processing transaction: {transaction.get('transaction_id')}")
            logger.info(f"   User: {transaction.get('user_id')}, Amount: ${transaction.get('amount', 0):.2f}")
            
            if self.decision_engine:
                # Apply fraud detection
                result = self.decision_engine.process_transaction(transaction)
                
                decision = result['decision']
                fraud_prob = result['fraud_probability']
                latency = result['latency_ms']
                amount = transaction.get('amount', 0)
                
                # Log decision with emoji
                if decision == "REJECT":
                    logger.error(
                        f"⛔ REJECT - Transaction: {result['transaction_id']}, "
                        f"Fraud Prob: {fraud_prob:.4f}, Latency: {latency:.2f}ms, "
                        f"Amount: ${amount:.2f}"
                    )
                elif decision == "PEND":
                    logger.warning(
                        f"⏸️  PEND - Transaction: {result['transaction_id']}, "
                        f"Fraud Prob: {fraud_prob:.4f}, Latency: {latency:.2f}ms, "
                        f"Amount: ${amount:.2f}"
                    )
                else:
                    logger.info(
                        f"✅ ALLOW - Transaction: {result['transaction_id']}, "
                        f"Fraud Prob: {fraud_prob:.4f}, Latency: {latency:.2f}ms, "
                        f"Amount: ${amount:.2f}"
                    )
                
                # Save decision to database
                self.db.save_decision(transaction, result)
                
                # Send alerts for REJECT/PEND
                if decision in ['REJECT', 'PEND']:
                    self.send_alert(result, transaction)
                
                # Update statistics
                self.stats.update(decision, amount, latency)
                
                # Update database statistics every 10 transactions
                self.stats_update_counter += 1
                if self.stats_update_counter >= 10:
                    summary = self.stats.get_summary()
                    self.db.update_statistics(summary)
                    logger.info(f"📊 Statistics: {summary}")
                    self.stats_update_counter = 0
                
            else:
                logger.warning("⚠️  Decision engine not available, skipping fraud detection")
                
        except Exception as e:
            logger.error(f"❌ Error processing transaction: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def consume_messages(self):
        """Continuously consume messages from Kafka"""
        self.running = True
        logger.info('🚀 Starting to consume messages...')
        
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info(f'📍 Reached end of partition {msg.partition()}')
                    else:
                        logger.error(f'❌ Consumer error: {msg.error()}')
                        raise KafkaException(msg.error())
                else:
                    try:
                        transaction = json.loads(msg.value().decode('utf-8'))
                        self.process_transaction(transaction)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f'❌ Failed to decode message: {str(e)}')
                    except Exception as e:
                        logger.error(f'❌ Error handling message: {str(e)}')
        
        except KeyboardInterrupt:
            logger.info('⚠️  Interrupted by user')
        finally:
            self.shutdown()
    
    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        if self.running:
            logger.info('🛑 Shutting down consumer...')
            self.running = False
            
            # Final statistics update
            summary = self.stats.get_summary()
            self.db.update_statistics(summary)
            logger.info(f"📊 Final Statistics: {summary}")
            
            if self.consumer:
                self.consumer.close()
                logger.info('✅ Consumer closed successfully')
            
            self.db.close()


if __name__ == "__main__":
    consumer = TransactionConsumer()
    consumer.consume_messages()