/**
 * Zod schemas mirror src/serving/schemas.py. A mismatch here is a 422 at runtime,
 * so these tests check the contract, not the convenience.
 */
import { describe, expect, it } from "vitest";

import {
  CHURN_DEFAULTS,
  ChurnInputSchema,
  CREDIT_RISK_DEFAULTS,
  CreditRiskInputSchema,
  FRAUD_DEFAULTS,
  FraudInputSchema,
} from "@/lib/schemas";

describe("defaults", () => {
  it("every set of defaults validates against its own schema", () => {
    // The forms initialise from these, so an invalid default is a broken page.
    expect(FraudInputSchema.safeParse(FRAUD_DEFAULTS).success).toBe(true);
    expect(CreditRiskInputSchema.safeParse(CREDIT_RISK_DEFAULTS).success).toBe(true);
    expect(ChurnInputSchema.safeParse(CHURN_DEFAULTS).success).toBe(true);
  });
});

describe("field names come from the real datasets", () => {
  it("fraud uses IEEE-CIS column names", () => {
    const keys = Object.keys(FRAUD_DEFAULTS);
    for (const column of ["TransactionAmt", "TransactionDT", "ProductCD", "card1", "P_emaildomain"]) {
      expect(keys).toContain(column);
    }
  });

  it("credit risk uses LendingClub column names", () => {
    const keys = Object.keys(CREDIT_RISK_DEFAULTS);
    for (const column of ["loan_amnt", "fico_range_low", "revol_util", "sub_grade"]) {
      expect(keys).toContain(column);
    }
  });

  it("credit risk exposes NO post-origination field", () => {
    // These are the columns that make a LendingClub model report 0.99 and be
    // worthless. They are denylisted server-side; they must not exist here at all.
    const keys = Object.keys(CREDIT_RISK_DEFAULTS);
    for (const leak of [
      "recoveries",
      "collection_recovery_fee",
      "total_rec_prncp",
      "total_pymnt",
      "last_pymnt_amnt",
      "out_prncp",
      "debt_settlement_flag",
      "loan_status",
    ]) {
      expect(keys).not.toContain(leak);
    }
  });

  it("churn exposes neither Naive_Bayes_Classifier column", () => {
    // Those two columns are the target laundered through a classifier.
    const keys = Object.keys(CHURN_DEFAULTS);
    expect(keys.some((k) => k.startsWith("Naive_Bayes_Classifier"))).toBe(false);
    expect(keys).not.toContain("CLIENTNUM");
    expect(keys).not.toContain("Attrition_Flag");
  });
});

describe("validation rejects impossible input", () => {
  it("rejects a non-positive transaction amount", () => {
    expect(
      FraudInputSchema.safeParse({ ...FRAUD_DEFAULTS, TransactionAmt: 0 }).success,
    ).toBe(false);
  });

  it("rejects a FICO score outside the real range", () => {
    expect(
      CreditRiskInputSchema.safeParse({ ...CREDIT_RISK_DEFAULTS, fico_range_low: 250 }).success,
    ).toBe(false);
  });

  it("rejects a malformed LendingClub sub-grade", () => {
    expect(
      CreditRiskInputSchema.safeParse({ ...CREDIT_RISK_DEFAULTS, sub_grade: "Z9" }).success,
    ).toBe(false);
  });

  it("rejects a utilisation ratio above 1", () => {
    expect(
      ChurnInputSchema.safeParse({ ...CHURN_DEFAULTS, Avg_Utilization_Ratio: 1.5 }).success,
    ).toBe(false);
  });

  it("rejects more than twelve inactive months in a twelve-month window", () => {
    expect(
      ChurnInputSchema.safeParse({ ...CHURN_DEFAULTS, Months_Inactive_12_mon: 13 }).success,
    ).toBe(false);
  });
});
