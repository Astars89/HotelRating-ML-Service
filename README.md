# Hotel Rating Prediction Model

This project is focused on developing a hotel rating prediction model using machine learning techniques. The model takes various features of a hotel as input and predicts its rating. The goal is to provide a tool that can help users make more informed decisions when choosing a hotel.

## Model Overview

- **Model Type**: Regression
- **Algorithm**: Random Forest Regressor
- **Features**: Multiple hotel-related features, including average score, review sentiment analysis, country, month, and more.
- **Data Source**: A dataset with information about hotels and their ratings.

## Dataset

The dataset [hotels.csv](https://www.kaggle.com/datasets/lidiyacutie/hotels) used for this project contains information on various hotels, including features like the average score, the number of reviews, reviewer scores, and more. It also includes information about the reviewers' nationality and the distance of the hotel from the city center.

## Jupyter Notebooks

In this project, Jupyter notebooks are included for various tasks, such as data exploration, model training, and results evaluation. You can find these notebook in the directory. Below is a list of available notebook and their descriptions:

1. `LSML2_Final_Project.ipynb`: This notebook contains code for exploring and visualizing the data used in the project and to train machine learning models and save them for future use.

## Model Training

The model was trained using a Random Forest Regressor, which is a powerful machine learning algorithm for regression tasks. The training data consisted of a subset of the dataset, with features and corresponding hotel ratings.

## How to Use the Model

1. **Run the Web Application**: To use the model, run the provided web application using Flask.

   ```
   python app.py
   ```

2. **Access the Web Interface**: Open a web browser and go to `http://localhost:5000` to access the web interface.

3. **Input Features**: Enter the required features in the input form on the web page. These features include the hotel's average score, the number of negative and positive words in reviews, the number of reviews, and others.

4. **Submit Data**: Click the "Submit" button to send the data to the model.

5. **Receive Prediction**: The model will return a predicted rating for the hotel based on the input features.

## Project Structure

- `app.py`: The Flask application for the web interface and model prediction.
- `model.pkl`: The trained Random Forest Regressor model saved using joblib.
- `templates`: Contains HTML templates for the web application.
- `static`: Contains static files like CSS and JavaScript for the web interface.

## Dependencies

- Python 3.x
- Flask
- scikit-learn
- joblib
- pandas
- HTML/CSS/JavaScript

## Running with Docker

If you wish to run this application using Docker, follow these steps:

1. Make sure you have Docker installed. If not, you can download it from the [official Docker website](https://docs.docker.com/get-docker/).

2. Build a Docker image using the provided Dockerfile in this repository. Run the following command from the root directory of your project:

   ```bash
   docker build -t my-hotel-predictor .


## Further Development

This project can be expanded by including more features, fine-tuning the model, and enhancing the web interface. Additionally, it can be deployed to a production environment to provide a real-time hotel rating prediction service to users.

For questions or improvements, please feel free to reach out. Enjoy using the Hotel Rating Prediction Model!

---

