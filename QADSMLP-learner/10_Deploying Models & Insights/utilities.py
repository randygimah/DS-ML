import joblib
import pandas as pd

def load_model():
	return joblib.load('model.joblib')

def load_encoders():
	return joblib.load('nominal.joblib'), joblib.load('ordinal.joblib')

def load_feature_names():
	return joblib.load('feature_names.joblib')

def format_data(data: dict):
	df = pd.DataFrame(data)
	nominal_encoder, ordinal_encoder = load_encoders()
	feature_names = load_feature_names()
	
	# Transform categorical columns
	df_nominal = nominal_encoder.transform(df[['destination', 'fare']])
	df_nominal_cols = nominal_encoder.get_feature_names_out(['destination', 'fare'])
	
	df_ordinal = ordinal_encoder.transform(df[['vehicle_class']])
	
	# Drop original categorical columns
	df = df.drop(['destination', 'fare', 'vehicle_class'], axis=1)
	
	# Add encoded columns as a DataFrame
	nominal_df = pd.DataFrame(df_nominal, columns=df_nominal_cols)
	ordinal_df = pd.DataFrame(df_ordinal, columns=['vehicle_class'])
	
	# Combine all DataFrames
	result = pd.concat([df, nominal_df, ordinal_df], axis=1)
	
	# Ensure columns are in the same order as training
	result = result[feature_names]
	
	return result

def make_prediction(data):
	if 'model' not in locals():
		model = load_model()
	return model.predict(format_data(data))


if __name__ == '__main__':
	main()