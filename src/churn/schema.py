from pandera.pandas import Check, Column, DataFrameSchema

YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["Yes", "No", "No internet service"]

schema = DataFrameSchema(
    {
        "customerID": Column(nullable=False, unique=True),
        "gender": Column(checks=Check.isin(["Female", "Male"])),
        "SeniorCitizen": Column(int, checks=Check.isin([0, 1])),
        "Partner": Column(checks=Check.isin(YES_NO)),
        "Dependents": Column(checks=Check.isin(YES_NO)),
        "tenure": Column(int, checks=Check.in_range(0, 72)),
        "PhoneService": Column(checks=Check.isin(YES_NO)),
        "MultipleLines": Column(checks=Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(checks=Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(checks=Check.isin(INTERNET_ADDON)),
        "OnlineBackup": Column(checks=Check.isin(INTERNET_ADDON)),
        "DeviceProtection": Column(checks=Check.isin(INTERNET_ADDON)),
        "TechSupport": Column(checks=Check.isin(INTERNET_ADDON)),
        "StreamingTV": Column(checks=Check.isin(INTERNET_ADDON)),
        "StreamingMovies": Column(checks=Check.isin(INTERNET_ADDON)),
        "Contract": Column(checks=Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(checks=Check.isin(YES_NO)),
        "PaymentMethod": Column(
            checks=Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            )
        ),
        "MonthlyCharges": Column(float, checks=Check.in_range(0, 200)),
        "TotalCharges": Column(checks=Check.str_matches(r"^\s*\d*\.?\d*\s*$")),
        "Churn": Column(checks=Check.isin(YES_NO)),
    },
    strict=True,  
)


def validate_raw(df):
    return schema.validate(df, lazy=True)