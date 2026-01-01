from src.entity import DataIngestionConfig
from src.entity import ModelTrainerConfig
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml,create_directories


class ConfigurationManager:
    def __init__(self,config_filepath= CONFIG_FILE_PATH,params_filepath= PARAMS_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    # Attribute for data ingestion
    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion                 # Extracts only the data_ingestion part of config.yaml.

        create_directories([config.root_dir])               # Create data_ingestion.root directory

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_model_trainer(self) -> ModelTrainerConfig:
        config = self.config.model_trainer                 # Extracts only the data_ingestion part of config.yaml.
        params = self.params.model_trainer
        create_directories([config.root_dir])               # Create data_ingestion.root directory

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            val_data_path=config.val_data_path,
            epochs= params.epochs,
            batch_size=params.batch_size
            
            
        )

        return model_trainer_config