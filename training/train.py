import os
import mlflow
import mlflow.tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from datetime import datetime

def log_to_db(db_url, model_name, version, accuracy):
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO model_training_logs 
            (model_name, version, accuracy, training_time) 
            VALUES (:name, :version, :accuracy, :time)
        """), {
            'name': model_name,
            'version': version,
            'accuracy': accuracy,
            'time': datetime.now()
        })
        conn.commit()

def create_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    model = create_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # MLflow setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("MNIST")

    with mlflow.start_run():
        # Train model
        model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test)
        
        # Log parameters and metrics
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 64)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        
        # Log model
        mlflow.tensorflow.log_model(
            model,
            "mnist_cnn",
            registered_model_name="mnist_cnn"
        )
        
        # Log to database
        if "DATABASE_URL" in os.environ:
            log_to_db(
                os.environ["DATABASE_URL"],
                "mnist_cnn",
                "1.0",
                float(test_acc)
            )

if __name__ == "__main__":
    main()