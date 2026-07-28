"""Request schemas for the three prediction endpoints.

Every field name is a **real column from the real dataset**, not a friendlier
invention. That matters: a reviewer reading ``/predict/fraud`` should be able to
open the IEEE-CIS data description and find ``card1`` and ``ProductCD`` there. It
also means the serving path can hand the payload straight to the same feature code
training used, with no translation layer to drift out of sync.

These are deliberate *subsets*. IEEE-CIS has 394 columns; a form with 394 inputs is
not a demo. Unsupplied columns arrive at the model as NaN, which LightGBM handles
natively as "unknown"; see ``src/serving/preprocessing.py``.

Defaults are seeded demo values, never real customer data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """A payment, in IEEE-CIS column names.

    ``TransactionDT`` is the dataset's seconds-from-reference offset, not a Unix
    timestamp. The default corresponds to roughly day 100 of the dataset window.
    """

    TransactionDT: int = Field(8_640_000, description="Seconds from the dataset reference point")
    TransactionAmt: float = Field(149.99, gt=0, description="Transaction amount")
    ProductCD: str = Field("W", description="Product code: W, C, R, H or S")
    card1: float = Field(13926, description="Anonymised card identifier")
    card2: float | None = Field(361.0, description="Anonymised card attribute")
    card3: float = Field(150.0, description="Anonymised card attribute")
    card4: str = Field("visa", description="Card network")
    card5: float = Field(226.0, description="Anonymised card attribute")
    card6: str = Field("debit", description="debit or credit")
    addr1: float = Field(315.0, description="Anonymised billing region")
    addr2: float = Field(87.0, description="Anonymised billing country")
    dist1: float | None = Field(19.0, description="Anonymised distance")
    P_emaildomain: str = Field("gmail.com", description="Purchaser email domain")
    R_emaildomain: str | None = Field(None, description="Recipient email domain")
    C1: float = Field(1.0, description="Counting feature")
    C13: float = Field(1.0, description="Counting feature")
    C14: float = Field(1.0, description="Counting feature")
    D1: float | None = Field(14.0, description="Timedelta feature (days)")
    D15: float | None = Field(0.0, description="Timedelta feature (days)")


class LoanApplicationRequest(BaseModel):
    """A loan application, in LendingClub column names.

    Every field is available at origination. No post-origination field appears
    here or anywhere in the serving path. That is the leak
    ``configs/credit_risk.yaml``'s denylist exists to prevent.
    """

    loan_amnt: float = Field(15000, gt=0, description="Requested amount")
    funded_amnt: float = Field(15000, gt=0, description="Funded amount")
    term: str = Field(" 36 months", description="' 36 months' or ' 60 months'")
    int_rate: float = Field(13.56, description="Interest rate, percent")
    installment: float = Field(509.66, gt=0, description="Monthly payment")
    grade: str = Field("C", description="LendingClub grade A-G")
    sub_grade: str = Field("C1", description="LendingClub sub-grade, e.g. C1")
    emp_length: str = Field("5 years", description="Employment length as reported")
    home_ownership: str = Field("MORTGAGE", description="RENT, OWN or MORTGAGE")
    annual_inc: float = Field(72000, gt=0, description="Self-reported annual income")
    verification_status: str = Field("Source Verified", description="Income verification")
    purpose: str = Field("debt_consolidation", description="Stated loan purpose")
    addr_state: str = Field("CA", description="Two-letter state")
    dti: float = Field(18.24, description="Debt-to-income ratio")
    delinq_2yrs: float = Field(0, ge=0, description="Delinquencies in the last 2 years")
    fico_range_low: float = Field(695, description="Lower bound of the FICO band")
    fico_range_high: float = Field(699, description="Upper bound of the FICO band")
    inq_last_6mths: float = Field(1, ge=0, description="Credit inquiries, last 6 months")
    open_acc: float = Field(11, ge=0, description="Open credit lines")
    pub_rec: float = Field(0, ge=0, description="Derogatory public records")
    revol_bal: float = Field(14300, ge=0, description="Revolving balance")
    revol_util: float = Field(52.4, description="Revolving utilisation, percent")
    total_acc: float = Field(24, ge=0, description="Total credit lines")
    application_type: str = Field("Individual", description="Individual or Joint App")
    mort_acc: float = Field(1, ge=0, description="Mortgage accounts")
    pub_rec_bankruptcies: float = Field(0, ge=0, description="Public-record bankruptcies")


class CardholderRequest(BaseModel):
    """A credit-card customer, in the attrition dataset's column names.

    The two ``Naive_Bayes_Classifier_*`` columns from the published CSV are
    absent on purpose: they are the target laundered through a classifier, and
    they are denylisted in ``configs/churn.yaml``.
    """

    Customer_Age: float = Field(45, ge=18, description="Age in years")
    Gender: str = Field("M", description="M or F")
    Dependent_count: float = Field(2, ge=0, description="Number of dependents")
    Education_Level: str = Field("Graduate", description="Education level as reported")
    Marital_Status: str = Field("Married", description="Marital status as reported")
    Income_Category: str = Field("$60K - $80K", description="Income band as reported")
    Card_Category: str = Field("Blue", description="Blue, Silver, Gold or Platinum")
    Months_on_book: float = Field(36, ge=0, description="Relationship length in months")
    Total_Relationship_Count: float = Field(4, ge=0, description="Products held")
    Months_Inactive_12_mon: float = Field(2, ge=0, description="Inactive months, last 12")
    Contacts_Count_12_mon: float = Field(3, ge=0, description="Service contacts, last 12")
    Credit_Limit: float = Field(12000, gt=0, description="Credit limit")
    Total_Revolving_Bal: float = Field(1200, ge=0, description="Revolving balance")
    Avg_Open_To_Buy: float = Field(10800, ge=0, description="Average available credit")
    Total_Amt_Chng_Q4_Q1: float = Field(0.75, description="Spend change Q4 vs Q1")
    Total_Trans_Amt: float = Field(4400, ge=0, description="Total transaction amount")
    Total_Trans_Ct: float = Field(67, ge=0, description="Total transaction count")
    Total_Ct_Chng_Q4_Q1: float = Field(0.68, description="Count change Q4 vs Q1")
    Avg_Utilization_Ratio: float = Field(0.1, ge=0, le=1, description="Average utilisation")


#: Route path fragment -> (problem name, request schema).
ROUTES = {
    "fraud": ("fraud", TransactionRequest),
    "credit-risk": ("credit_risk", LoanApplicationRequest),
    "churn": ("churn", CardholderRequest),
}
