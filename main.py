from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        features = request.form['features'].split()
        file = request.files['files']
        target = request.form['target']

        file.save('static/' + file.filename)
        iowa_file_path = 'static/' + file.filename
        home_data = pd.read_csv(iowa_file_path)

        q1 = home_data[target].quantile(0.25)
        q3 = home_data[target].quantile(0.75)

        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        new_data = pd.DataFrame(home_data[(home_data["SalePrice"] > lower_limit) & (home_data["SalePrice"] < upper_limit)])
        new_data = new_data.reset_index(drop=False, inplace=False)
        print(new_data)
        y = new_data[target]

       # Create X
        X = new_data[features]

        # Split into validation and training data
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

        # Define a random forest model
        rf_model = RandomForestRegressor(random_state=1)
        rf_model.fit(train_X, train_y)
        rf_val_predictions = rf_model.predict(val_X)
        rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

        print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

        # To improve accuracy, create a new Random Forest model which you will train on all training data
        rf_model_on_full_data = RandomForestRegressor()

        # Fit rf_model_on_full_data on all data from the training data
        rf_model_on_full_data.fit(X, y)

        # Read test data file using pandas



        # Create test_X which includes only the columns used for prediction
        test_X = new_data[features]

        # Make predictions
        test_preds = rf_model_on_full_data.predict(test_X)

        # Create DataFrame for predictions
        output = pd.DataFrame({'Id': new_data['Id'], target: test_preds})




        data = {'features': output.to_html(index=False)}

        return render_template('index.html', data=data)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
