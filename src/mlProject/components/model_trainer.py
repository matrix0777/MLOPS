import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load datasets
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Split features and target
        train_x = train_data.drop(columns=[self.config.target_column])
        test_x = test_data.drop(columns=[self.config.target_column])
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Train model
        model = ElasticNet(
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            random_state=42
        )
        model.fit(train_x, train_y)

        # Save model
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
