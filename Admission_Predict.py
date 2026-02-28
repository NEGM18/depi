import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Admission_Predict.csv')
df.columns = df.columns.astype(str).str.strip()
target_col = 'Chance of Admit'

if target_col not in df.columns:
    print(f"Error: Target column '{target_col}' not found.")
    print("Available columns:", df.columns.tolist())
else:
    X = df.drop(columns=['Serial No.', target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_depth = 5
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Decision Tree Regression Model Performance (max_depth={max_depth}):")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['Importance']
    ).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importances)
    print("\nVisualizing the Decision Tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=X.columns.tolist(), 
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title("Decision Tree for Admission Prediction")
    output_image = "decision_tree.png"
    plt.savefig(output_image, bbox_inches='tight')
    print(f"Decision tree visualization saved as '{output_image}'")

    plt.show()
