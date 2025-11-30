from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import from train module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import load_data, evaluate
from utils.results import save_results


def linear_regression(X_train, y_train, X_test, y_test):
    """
    Train a linear regression model and evaluate it on training and test sets.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target values
        X_test: Test feature matrix
        y_test: Test target values
        
    Returns:
        Tuple of (model, RMSE_Train, R2_Train, RMSE_Test, R2_Test)
    """
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_hat_train = model.predict(X_train)
    RMSE_Train, R2_Train = evaluate(y_train, y_hat_train, 'train')
    y_hat_test = model.predict(X_test)
    RMSE_Test, R2_Test = evaluate(y_test, y_hat_test, 'test')

    # Save results to centralized results.csv
    save_results(
        model_name='LinearRegression',
        rmse_train=RMSE_Train,
        r2_train=R2_Train,
        rmse_test=RMSE_Test,
        r2_test=R2_Test
    )

    return model, RMSE_Train, R2_Train, RMSE_Test, R2_Test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data('../data/processed')
    model, RMSE_Train, R2_Train, RMSE_Test, R2_Test = linear_regression(X_train, y_train, X_test, y_test)
    print(f"RMSE_Train: {RMSE_Train}, R2_Train: {R2_Train}, RMSE_Test: {RMSE_Test}, R2_Test: {R2_Test}")