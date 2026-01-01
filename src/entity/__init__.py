from dataclasses import dataclass
from pathlib import Path

# Configuration for data ingestion

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_url :str
    local_data_file :Path
    unzip_dir :Path


@dataclass
class ModelTrainerConfig:
  root_dir : Path
  train_data_path : Path
  val_data_path : Path
  epochs: int
  batch_size: int
  