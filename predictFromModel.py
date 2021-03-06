import pandas as pd
import numpy as np
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation


class prediction:

    def __init__(self, path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile()  # deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object, 'Start of Prediction')
            data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
            data = data_getter.get_data()

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            data = preprocessor.remove_columns(data,'Datetime')

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation

            data = preprocessor.encode_categorical_columns(data)
            print(data.head(10))
            # Proceeding with more data pre-processing steps


            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)

            model_name = 'XGBoost1'
            model = file_loader.load_model(model_name)
            result = model.predict(data)

            final = pd.DataFrame(result,columns=['Predictions'])
            path = "Prediction_Output_File/Predictions.csv"
            final.to_csv(path, index=False)  # appends result to prediction file
            self.log_writer.log(self.file_object, 'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: :: %s' % ex)
            raise ex
        return path




