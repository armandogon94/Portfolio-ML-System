"use client";

/**
 * Payment-fraud demo page.
 *
 * The inputs are IEEE-CIS's real column names. The output is a probability, a
 * review-queue band, and the git SHA of the commit that trained the checkpoint.
 */

import { zodResolver } from "@hookform/resolvers/zod";

import { ModelPage, type ResultView } from "@/components/ModelPage";
import type { Tone } from "@/components/PredictionResult";
import {
  explainFraud,
  predictFraud,
  type FraudPrediction,
  type FraudRiskBand,
} from "@/lib/api";
import { FraudInputSchema, FRAUD_DEFAULTS, type FraudInput } from "@/lib/schemas";
import { FRAUD_FIELDS } from "./fields";

/** Band -> visual tone. LOW is benign; CRITICAL means decline. */
const BAND_TONE: Record<FraudRiskBand, Tone> = {
  LOW: "good",
  MEDIUM: "warn",
  HIGH: "warn",
  CRITICAL: "bad",
};

function toResultView(prediction: FraudPrediction): ResultView {
  return {
    decisionLabel: "Recommended action",
    decision: prediction.recommended_action,
    decisionTone: BAND_TONE[prediction.risk_band],
    probabilityLabel: "Fraud probability",
    probability: prediction.fraud_probability,
    modelVersion: prediction.model_version,
    trainedOn: prediction.trained_on,
    caveat:
      "Band thresholds are a readable convention, not a calibrated policy. " +
      "Calibrating them needs the cost of a false decline against a chargeback.",
  };
}

export default function FraudPage() {
  return (
    <ModelPage<FraudInput, FraudPrediction>
      title="Payment Fraud Detection"
      intro="Score a payment against a LightGBM model trained on IEEE-CIS (Vesta) real e-commerce transactions with a time-based split."
      resolver={zodResolver(FraudInputSchema)}
      defaults={FRAUD_DEFAULTS}
      fields={FRAUD_FIELDS}
      predict={predictFraud}
      explain={explainFraud}
      toResultView={toResultView}
    />
  );
}
