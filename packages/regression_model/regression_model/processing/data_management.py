Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@zahid1202 
Learn Git and GitHub without any code!
Using the Hello World guide, you’ll start a branch, write comments, and open a pull request.


0
0839zahid1202/deploying-machine-learning-models
forked from trainindata/deploying-machine-learning-models
 Code Pull requests 0 Actions Projects 0 Wiki Security Insights Settings
deploying-machine-learning-models/packages/regression_model/regression_model/processing/data_management.py
@ChristopherGS ChristopherGS Section 9.3 - Differential Tests in CI Part 1
5a0382f on Jan 26
61 lines (44 sloc)  1.83 KB
  
Code navigation is available!
Navigate your code with ease. Click on function and method calls to jump to their definitions or references in the same repository. Learn more

 Code navigation is available for this repository but data for this commit does not exist.

Learn more or give us feedback
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from regression_model.config import config
from regression_model import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f'saved pipeline: {save_file_name}')


def load_pipeline(*, file_name: str
                  ) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

