"use client";

/**
 * Consumer-credit-risk demo page.
 *
 * Inputs are LendingClub origination-time fields only. The response carries the
 * API's own statement that the approve/review/decline cut points are
 * illustrative rather than a credit policy, and that text is rendered as-is.
 */

import { zodResolver } from "@hookform/resolvers/zod";

import { ModelPage, type ResultView } from "@/components/ModelPage";
import type { Tone } from "@/components/PredictionResult";
import {
  explainCreditRisk,
  predictCreditRisk,
  type CreditDecision,
  type CreditRiskPrediction,
} from "@/lib/api";
import {
  CreditRiskInputSchema,
  CREDIT_RISK_DEFAULTS,
  type CreditRiskInput,
} from "@/lib/schemas";
import { CREDIT_RISK_FIELDS } from "./fields";

const DECISION_TONE: Record<CreditDecision, Tone> = {
  APPROVE: "good",
  REVIEW: "warn",
  DECLINE: "bad",
};

function toResultView(prediction: CreditRiskPrediction): ResultView {
  return {
    decisionLabel: "Decision",
    decision: prediction.decision,
    decisionTone: DECISION_TONE[prediction.decision],
    probabilityLabel: "Default probability",
    probability: prediction.default_probability,
    modelVersion: prediction.model_version,
    trainedOn: prediction.trained_on,
    caveat: prediction.threshold_basis,
  };
}

export default function CreditRiskPage() {
  return (
    <ModelPage<CreditRiskInput, CreditRiskPrediction>
      title="Consumer Credit Risk"
      intro="Score a loan application against a LightGBM model trained on LendingClub 2007-2018Q4, split by issue date, with every post-origination field denylisted."
      resolver={zodResolver(CreditRiskInputSchema)}
      defaults={CREDIT_RISK_DEFAULTS}
      fields={CREDIT_RISK_FIELDS}
      predict={predictCreditRisk}
      explain={explainCreditRisk}
      toResultView={toResultView}
    />
  );
}
