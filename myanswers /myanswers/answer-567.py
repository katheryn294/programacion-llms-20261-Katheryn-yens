from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def pipeline_svr(df, target_col, test_size):
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 4)

    return y_pred, mae
